/**
 * =============================================================================
 * TypeScript Image Search CLI using OpenAI's CLIP Model (v3.4.0)
 * =============================================================================
 *
 * Description:
 * This script is a self-contained command-line tool for macOS that performs
 * semantic image search. It uses the `@xenova/transformers` library to run a
 * quantized version of the CLIP model directly in Node.js.
 *
 * NEW in v3.4.0: Added a similarity threshold to only return meaningful matches.
 *
 * To set up (run once):
 * 1. mkdir image-search-ts && cd image-search-ts
 * 2. npm init -y
 * 3. npm install @xenova/transformers
 * 4. npm install -D typescript ts-node @types/node
 * 5. npx tsc --init --rootDir ./ --outDir ./dist --esModuleInterop --resolveJsonModule --module nodenext --moduleResolution nodenext --target es2022
 * 6. Create this file as `search.ts`
 *
 * To run:
 * - Indexing: npx ts-node search.ts index ./path/to/your/images [limit]
 * - Searching: npx ts-node search.ts search "your search query" [optional_top_n]
 *
 */

import {
    AutoTokenizer,
    AutoProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    RawImage,
} from '@xenova/transformers';

import * as fs from 'fs/promises';
import * as path from 'path';
import { exec } from 'child_process';

// --- CONFIGURATION ---
const SCRIPT_VERSION = 'v3.4.0';
const MODEL_NAME = 'Xenova/clip-vit-base-patch32';
const DB_PATH = './db.json';
const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
const SIMILARITY_THRESHOLD = 0.28; // Only show results with a score >= this value

// Define the structure for our simple database entry.
interface DbEntry {
  filePath: string;
  embedding: number[];
}

/**
 * A singleton class to manage the CLIP model and processor instances.
 * This ensures the heavy models are loaded only once.
 */
class CLIPSingleton {
  private static tokenizer: any | null = null;
  private static processor: any | null = null;
  private static visionModel: CLIPVisionModelWithProjection | null = null;
  private static textModel: CLIPTextModelWithProjection | null = null;

  static async getTokenizer() {
    if (this.tokenizer === null) {
      console.log('Loading tokenizer...');
      this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
      console.log('Tokenizer loaded.');
    }
    return this.tokenizer;
  }

  static async getProcessor() {
    if (this.processor === null) {
      console.log('Loading image processor...');
      this.processor = await AutoProcessor.from_pretrained(MODEL_NAME);
      console.log('Image processor loaded.');
    }
    return this.processor;
  }

  static async getVisionModel() {
    if (this.visionModel === null) {
      console.log('Loading CLIP Vision model... This may take a moment.');
      this.visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true });
      console.log('Vision model loaded.');
    }
    return this.visionModel;
  }

  static async getTextModel() {
    if (this.textModel === null) {
      console.log('Loading CLIP Text model... This may take a moment.');
      this.textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true });
      console.log('Text model loaded.');
    }
    return this.textModel;
  }
}


/**
 * Calculates the cosine similarity between two vectors.
 * This is the core of our search algorithm.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @returns The cosine similarity score (between -1 and 1).
 */
function cosineSimilarity(v1: number[], v2: number[]): number {
  const dotProduct = v1.reduce((sum, a, i) => sum + a * v2[i], 0);
  const magnitude1 = Math.sqrt(v1.reduce((sum, a) => sum + a * a, 0));
  const magnitude2 = Math.sqrt(v2.reduce((sum, a) => sum + a * a, 0));

  if (magnitude1 === 0 || magnitude2 === 0) {
    return 0; // Avoid division by zero
  }

  return dotProduct / (magnitude1 * magnitude2);
}

/**
 * Generates an embedding for a given image file.
 * @param visionModel The CLIP vision model.
 * @param processor The image processor.
 * @param filePath The path to the image.
 * @returns A feature vector (embedding).
 */
async function embedImage(visionModel: any, processor: any, filePath: string): Promise<number[]> {
  const image = await RawImage.read(filePath);
  const image_inputs = await processor(image);
  const { image_embeds } = await visionModel(image_inputs);
  return Array.from(image_embeds.data);
}

/**
 * Generates an embedding for a given text query.
 * @param textModel The CLIP text model.
 * @param tokenizer The text tokenizer.
 * @param text The text to embed.
 * @returns A feature vector (embedding).
 */
async function embedText(textModel: any, tokenizer: any, text: string): Promise<number[]> {
  const text_inputs = tokenizer(text, { padding: true, truncation: true });
  const { text_embeds } = await textModel(text_inputs);
  return Array.from(text_embeds.data);
}

/**
 * Opens a file using the default system application.
 * @param filePath The path to the file to open.
 */
function openFile(filePath: string) {
  const command = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'start' : 'xdg-open';
  exec(`"${command}" "${filePath}"`, (error) => {
    if (error) {
      console.error(`❌ Failed to open file: ${error.message}`);
      return;
    }
    console.log(`✅ Opened ${path.basename(filePath)}`);
  });
}

/**
 * Recursively finds all image files in a given directory.
 * @param dir The directory to start scanning from.
 * @returns An array of full file paths to images.
 */
async function findImageFilesRecursive(dir: string): Promise<string[]> {
  let imageFiles: string[] = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      imageFiles = imageFiles.concat(await findImageFilesRecursive(fullPath));
    } else if (IMAGE_EXTENSIONS.includes(path.extname(entry.name).toLowerCase())) {
      imageFiles.push(fullPath);
    }
  }

  return imageFiles;
}

/**
 * INDEXING FUNCTION
 * Scans a directory for images, generates embeddings for them,
 * and saves them to our JSON database file.
 * @param imagesDir The directory containing images to index.
 * @param limit An optional limit on the number of images to process.
 */
async function indexImages(imagesDir: string, limit?: number) {
  console.log('🚀 Starting image indexing (recursive search)...');
  if (limit) {
    console.log(`⚠️  Processing limit set to ${limit} images.`);
  }

  const visionModel = await CLIPSingleton.getVisionModel();
  const processor = await CLIPSingleton.getProcessor();
  const db: DbEntry[] = [];
  let processedCount = 0;

  try {
    const imageFiles = await findImageFilesRecursive(imagesDir);
    const totalFiles = imageFiles.length;
    console.log(`Found ${totalFiles} total images to process.`);

    if (totalFiles === 0) return;

    for (const filePath of imageFiles) {
      if (limit && processedCount >= limit) {
        console.log(`Reached processing limit of ${limit}. Stopping.`);
        break;
      }

      processedCount++;
      console.log(`  [${processedCount}/${totalFiles}] Processing ${filePath}...`);
      try {
        const embedding = await embedImage(visionModel, processor, filePath);
        db.push({ filePath, embedding });
      } catch (e) {
        console.log(`  Skipping file ${filePath} due to an error: ${(e as Error).message}`);
      }
    }

    await fs.writeFile(DB_PATH, JSON.stringify(db, null, 2));
    console.log(`✅ Indexing complete! Database with ${db.length} entries saved to ${DB_PATH}`);
  } catch (error) {
    console.error(`❌ Error during indexing: ${(error as Error).message}`);
    console.error(`Please ensure the directory "${imagesDir}" exists.`);
  }
}


/**
 * SEARCH FUNCTION
 * Takes a text query, embeds it, and finds the most similar
 * images from our database, then opens them.
 * @param query The text to search for.
 * @param topN The number of top results to return.
 */
async function search(query: string, topN: number = 4) {
  console.log(`🔎 Searching for: "${query}" (Top ${topN})`);

  // 1. Ensure the database exists
  let db: DbEntry[];
  try {
    const dbFile = await fs.readFile(DB_PATH, 'utf-8');
    db = JSON.parse(dbFile);
  } catch (error) {
    console.error(`❌ Database file not found at ${DB_PATH}.`);
    console.error('Please run the "index" command first.');
    return;
  }

  if (db.length === 0) {
    console.error('❌ The database is empty. Please index some images first.');
    return;
  }

  // 2. Get the model and generate a text embedding for the query
  const textModel = await CLIPSingleton.getTextModel();
  const tokenizer = await CLIPSingleton.getTokenizer();
  const queryEmbedding = await embedText(textModel, tokenizer, query);

  // 3. Find all matches and sort them
  const allMatches = db.map(entry => {
    const similarity = cosineSimilarity(queryEmbedding, entry.embedding);
    return { entry, similarity };
  });
  allMatches.sort((a, b) => b.similarity - a.similarity);

  // 4. Filter matches by the similarity threshold
  const meaningfulMatches = allMatches.filter(match => match.similarity >= SIMILARITY_THRESHOLD);

  // 5. Get the top N results from the meaningful matches
  const topMatches = meaningfulMatches.slice(0, topN);


  // 6. Show and open the results
  if (topMatches.length > 0) {
    console.log(`\n✨ Top ${topMatches.length} meaningful matches found! ✨`);
    topMatches.forEach((match, index) => {
      console.log(`\n--- Result ${index + 1} ---`);
      console.log(`   File: ${match.entry.filePath}`);
      console.log(`   Similarity Score: ${(match.similarity * 100).toFixed(2)}%`);
      openFile(match.entry.filePath);
    });

  } else {
    console.log('\nCould not find any meaningful matches.');
    // Optionally, show the best but "not good enough" match for debugging/interest
    if (allMatches.length > 0) {
        console.log(`(The best match was "${path.basename(allMatches[0].entry.filePath)}" with a score of ${(allMatches[0].similarity * 100).toFixed(2)}%, which is below the threshold of ${(SIMILARITY_THRESHOLD * 100).toFixed(2)}%.)`);
    }
  }
}


/**
 * MAIN FUNCTION
 * Parses command-line arguments to decide whether to index or search.
 */
async function main() {
  console.log(`--- Image Search CLI ${SCRIPT_VERSION} ---`);
  const args = process.argv.slice(2);
  const command = args[0];
  const firstArg = args[1];

  if (command === 'index' && firstArg) {
    const limitArg = args[2] ? parseInt(args[2], 10) : undefined;
    await indexImages(firstArg, limitArg);
  } else if (command === 'search' && firstArg) {
    // Check if the last argument is a number for topN
    const lastArg = args[args.length - 1];
    let topN = 4; // Default value
    let queryArgs = args.slice(1);

    const parsedLastArg = parseInt(lastArg, 10);
    // Check if the last argument is a number and it's not the only part of the query
    if (!isNaN(parsedLastArg) && queryArgs.length > 1) {
        topN = parsedLastArg;
        queryArgs.pop(); // Remove the number from query arguments
    }
    const query = queryArgs.join(' ');
    await search(query, topN);

  } else {
    console.log('❌ Invalid command. Please use one of the following:');
    console.log('   To index images: npx ts-node search.ts index <path_to_folder> [optional_limit]');
    console.log('   To search:       npx ts-node search.ts search "your search query" [optional_top_n]');
  }
}

main().catch(console.error);

