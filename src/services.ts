import { GoogleGenAI, HarmCategory, HarmBlockThreshold, ContentEmbedding } from "@google/genai";
import { v4 as uuidv4 } from 'uuid';
import * as dotenv from 'dotenv';
import pdfParse from 'pdf-parse';
import { Pinecone } from '@pinecone-database/pinecone';

// Configure dotenv before any other imports that need env variables
dotenv.config();

// --- ENVIRONMENT VARIABLES ---
// IMPORTANT: Create a .env file in your root directory and add your API key.
// GOOGLE_API_KEY=your_api_key_here
const API_KEY = process.env.GOOGLE_API_KEY as string;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY as string;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME as string; // You'll need to add this to your .env

// --- MODEL & AI SETUP ---
const genAI = new GoogleGenAI({ apiKey: API_KEY });

const generationConfig = {
    temperature: 0.9,
    topK: 1,
    topP: 1,
    maxOutputTokens: 2048,
};

const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

// --- VECTOR DATABASE ---

let pinecone: Pinecone;

async function initPinecone() {
    if (!pinecone) {
        pinecone = new Pinecone({
            apiKey: PINECONE_API_KEY,
        });
    }
    return pinecone;
}

// --- UTILITY FUNCTIONS ---

/**
 * Splits a long text into smaller chunks of a specified size.
 * @param text The text to split.
 * @param chunkSize The desired size of each chunk.
 * @returns An array of text chunks.
 */
function chunkText(text: string, chunkSize = 1000): string[] {
    const chunks: string[] = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.substring(i, i + chunkSize));
    }
    return chunks;
}


// --- FILE PROCESSING & EMBEDDING ---

/**
 * Handles the upload of a file, chunks it, creates embeddings, and stores them in the vector DB.
 * @param fileBuffer The buffer containing the file data.
 * @param originalname The original name of the file.
 * @returns The ID of the document.
 */
export async function handleFileUpload(fileBuffer: Buffer, originalname: string): Promise<string> {
    console.log(`Processing file: ${originalname}`);
    const documentId = uuidv4();

    // 1. Read and chunk the text
    const text = await pdfParse(fileBuffer);
    console.log(text)
    const chunks = chunkText(text.text);
    console.log(`File split into ${chunks.length} chunks.`);

    // 2. Generate and store embeddings for each chunk in Pinecone
    const pineconeClient = await initPinecone();
    const index = pineconeClient.Index(PINECONE_INDEX_NAME);

    const vectorsToUpsert = [];

    for (const chunk of chunks) {
        const result = await genAI.models.embedContent(
            {
                model: "gemini-embedding-001",
                contents: { parts: [{ text: chunk }] },
                config: {
                    taskType: "RETRIEVAL_DOCUMENT",
                    outputDimensionality: 1024,
                },
            }
        );
        const embedding = result.embeddings[0].values;

        vectorsToUpsert.push({
            id: uuidv4(),
            values: embedding,
            metadata: { text: chunk, documentId: documentId, originalname: originalname },
        });
    }

    // Upsert vectors in batches (Pinecone recommends batches of 100)
    const batchSize = 100;
    for (let i = 0; i < vectorsToUpsert.length; i += batchSize) {
        const batch = vectorsToUpsert.slice(i, i + batchSize);
        await index.upsert(batch);
    }

    console.log(`Successfully created and stored ${vectorsToUpsert.length} vectors in Pinecone for document ${originalname}.`);
    return documentId;
}


// --- CHAT & RETRIEVAL-AUGMENTED GENERATION (RAG) ---

/**
 * Handles a chat query by searching the vector DB for context and generating a response.
 * @param query The user's question.
 * @param history The previous chat history.
 * @returns The generated response from the Gemini model.
 */
export async function handleChat(query: string, history: any[]): Promise<any> {
    console.log(`Handling chat query: "${query}"`);

    // 1. Embed the user's query
    const queryEmbeddingResult = await genAI.models.embedContent(
        {
            model: "gemini-embedding-001",
            contents: { parts: [{ text: query }] },
            config: {
                taskType: "RETRIEVAL_QUERY",
                outputDimensionality: 1024,
            }
        }
    );
    const queryEmbedding = queryEmbeddingResult.embeddings[0].values;

    // 2. Query Pinecone for similar documents
    const pineconeClient = await initPinecone();
    const index = pineconeClient.Index(PINECONE_INDEX_NAME);

    const queryResponse = await index.query({
        vector: queryEmbedding,
        topK: 5, // Get top 5 most similar chunks
        includeMetadata: true,
    });

    const similarDocs = queryResponse.matches?.map(match => match.metadata?.text).filter(Boolean) || [];

    const context = similarDocs.join(`

---

`);
    console.log(`Found ${similarDocs.length} relevant document chunks from Pinecone.`);

    const prompt = `
        Based on the following context, please answer the user's question.
        If the context does not contain the answer, say that you don't have enough information.

        CONTEXT:
        ${context}

        QUESTION:
        ${query}
    `;

    // 3. Prepare the prompt for the chat model
    const chatModel = await genAI.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
            ...generationConfig,
            safetySettings: safetySettings
        },
    });

    // 4. Generate the response

    console.log("Generated response successfully.");
    return chatModel.text;
}

/**
 * Handles a chat query by searching the vector DB for context and generating a response.
 * @param query The user's question.
 * @param history The previous chat history.
 * @returns The generated response from the Gemini model.
 */
export async function handleGenerate(query: string, history: any[]): Promise<string> {

    const chatModel = await genAI.models.generateContent({
        model: "gemini-2.5-flash",
        contents: query,
        config: {
            ...generationConfig,
            safetySettings: safetySettings
        },
    });

    return chatModel.text;
}
