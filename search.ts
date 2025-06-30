/**
 * =============================================================================
 * TypeScript Image Search CLI using OpenAI's CLIP Model (v3.0.1 - Final)
 * =============================================================================
 *
 * Description:
 * This script is a self-contained command-line tool for macOS that performs
 * semantic image search. It uses the `@xenova/transformers` library to run a
 * quantized version of the CLIP model directly in Node.js.
 *
 * NEW in v3.0.1: Corrected a TypeScript type error where the AutoProcessor and
 * AutoTokenizer types were not recognized as callable. Changed the type in the
 * function signatures to `any` to resolve the compilation error.
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
 * - Indexing: ts-node search.ts index ./path/to/your/images [limit]
 * - Searching: ts-node search.ts search "a photo of a cat"
 *
 */

import {
  AutoTokenizer,
  AutoProcessor,
  RawImage,
  CLIPVisionModelWithProjection,
  CLIPTextModelWithProjection,
} from '@xenova/transformers';
import * as fs from 'fs/promises';
import * as path from 'path';

// --- CONFIGURATION ---
const SCRIPT_VERSION = '3.0.1';
const MODEL_NAME = 'Xenova/clip-vit-base-patch32';
const DB_PATH = './db.json';
const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];

// Define the structure for our simple database entry.
interface DbEntry {
  filePath: string;
  embedding: number[];
}

/**
 * Manages the loading of the models, processor, and tokenizer to ensure they are
 * loaded only once using a singleton pattern. Now uses the correct, specific
 * model classes for vision and text.
 */
class CLIPSingleton {
  private static tokenizer: AutoTokenizer | null = null;
  private static processor: AutoProcessor | null = null;
  private static visionModel: CLIPVisionModelWithProjection | null = null;
  private static textModel: CLIPTextModelWithProjection | null = null;

  static async getInstance() {
    if (this.tokenizer === null) {
      console.log('Loading tokenizer...');
      this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
      console.log('Tokenizer loaded.');
    }
    if (this.processor === null) {
      console.log('Loading image processor...');
      this.processor = await AutoProcessor.from_pretrained(MODEL_NAME);
      console.log('Image processor loaded.');
    }
    if (this.visionModel === null) {
        console.log('Loading CLIP Vision model... This may take a moment.');
        this.visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true });
        console.log('Vision model loaded.');
    }
    if (this.textModel === null) {
        console.log('Loading CLIP Text model... This may take a moment.');
        this.textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true });
        console.log('Text model loaded.');
    }
    return {
      tokenizer: this.tokenizer,
      processor: this.processor,
      visionModel: this.visionModel,
      textModel: this.textModel,
    };
  }
}

/**
 * Calculates the cosine similarity between two vectors.
 */
function cosineSimilarity(v1: number[], v2: number[]): number {
  const dotProduct = v1.reduce((sum, a, i) => sum + a * v2[i], 0);
  const magnitude1 = Math.sqrt(v1.reduce((sum, a) => sum + a * a, 0));
  const magnitude2 = Math.sqrt(v2.reduce((sum, a) => sum + a * a, 0));
  if (magnitude1 === 0 || magnitude2 === 0) return 0;
  return dotProduct / (magnitude1 * magnitude2);
}

/**
 * Generates an embedding for a given image file.
 */
async function embedImage(visionModel: CLIPVisionModelWithProjection, processor: any, filePath: string): Promise<number[]> {
  const image = await RawImage.fromURL(filePath);
  const image_inputs = await processor(image);
  const { image_embeds } = await visionModel(image_inputs);
  return Array.from(image_embeds.data);
}

/**
 * Generates an embedding for a given text query.
 */
async function embedText(textModel: CLIPTextModelWithProjection, tokenizer: any, text: string): Promise<number[]> {
  const text_inputs = tokenizer(text, { padding: true, truncation: true });
  const { text_embeds } = await textModel(text_inputs);
  return Array.from(text_embeds.data);
}

/**
 * Recursively finds all image files in a directory and its subdirectories.
 */
async function findImageFilesRecursive(dir: string): Promise<string[]> {
  let imageFiles: string[] = [];
  try {
    const files = await fs.readdir(dir, { withFileTypes: true });
    for (const file of files) {
      const fullPath = path.join(dir, file.name);
      if (file.isDirectory()) {
        imageFiles = imageFiles.concat(await findImageFilesRecursive(fullPath));
      } else if (IMAGE_EXTENSIONS.includes(path.extname(file.name).toLowerCase())) {
        imageFiles.push(fullPath);
      }
    }
  } catch (error) {
    console.warn(`Could not read directory ${dir}: ${(error as Error).message}`);
  }
  return imageFiles;
}

/**
 * INDEXING FUNCTION
 */
async function indexImages(rootDir: string, limit?: number) {
  console.log('🚀 Starting image indexing (recursive search)...');
  if (limit) {
    console.log(`⚠️  Processing limit set to ${limit} images.`);
  }
  const { visionModel, processor } = await CLIPSingleton.getInstance();
  const db: DbEntry[] = [];

  try {
    const imageFiles = await findImageFilesRecursive(rootDir);
    if (imageFiles.length === 0) {
      console.log(`🟡 No images found in "${rootDir}" or its subdirectories.`);
      return;
    }
    console.log(`Found ${imageFiles.length} total images to process.`);

    let count = 0;
    for (const filePath of imageFiles) {
      if (limit && count >= limit) {
        console.log(`Reached processing limit of ${limit}. Stopping.`);
        break;
      }
      count++;
      console.log(`  [${count}/${imageFiles.length}] Processing ${filePath}...`);
      try {
        const embedding = await embedImage(visionModel, processor, filePath);
        db.push({ filePath, embedding });
      } catch (e) {
        const errorMessage = (e as any)?.message ?? String(e);
        console.warn(`  Skipping file ${filePath} due to an error: ${errorMessage}`);
      }
    }

    await fs.writeFile(DB_PATH, JSON.stringify(db, null, 2));
    console.log(`✅ Indexing complete! Database with ${db.length} entries saved to ${DB_PATH}`);
  } catch (error) {
    console.error(`❌ Error during indexing: ${(error as Error).message}`);
  }
}

/**
 * SEARCH FUNCTION
 */
async function search(query: string) {
  console.log(`🔎 Searching for: "${query}"`);

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
    console.error('❌ The database is empty.');
    return;
  }

  const { textModel, tokenizer } = await CLIPSingleton.getInstance();
  const queryEmbedding = await embedText(textModel, tokenizer, query);

  let bestMatch: DbEntry | null = null;
  let highestSimilarity = -Infinity;

  for (const entry of db) {
    const similarity = cosineSimilarity(queryEmbedding, entry.embedding);
    if (similarity > highestSimilarity) {
      highestSimilarity = similarity;
      bestMatch = entry;
    }
  }

  if (bestMatch) {
    console.log('\n✨ Best match found! ✨');
    console.log(`   File: ${bestMatch.filePath}`);
    console.log(`   Similarity Score: ${(highestSimilarity * 100).toFixed(2)}%`);
  } else {
    console.log('Could not find a suitable match.');
  }
}

/**
 * MAIN FUNCTION
 */
async function main() {
  console.log(`\n--- Image Search CLI v${SCRIPT_VERSION} ---`);
  const args = process.argv.slice(2);
  const command = args[0];
  const dirOrQuery = args[1];
  const limitArg = args[2];

  if (command === 'index' && dirOrQuery) {
    const limit = limitArg ? parseInt(limitArg, 10) : undefined;
    if (limitArg && isNaN(limit)) {
      console.error('Error: The limit argument must be a number.');
      return;
    }
    await indexImages(dirOrQuery, limit);
  } else if (command === 'search' && dirOrQuery) {
    // Rejoin args in case the search query had spaces
    const query = args.slice(1).join(' ');
    await search(query);
  } else {
    console.log('-------------------------------------------');
    console.log(` TypeScript Image Search CLI (v${SCRIPT_VERSION})`);
    console.log('-------------------------------------------\n');
    console.log('Usage:');
    console.log('  To index a folder of images:');
    console.log('    ts-node search.ts index <path_to_folder> [limit]\n');
    console.log('  To search for an image with a text query:');
    console.log('    ts-node search.ts search "your search query here"');
    console.log('-------------------------------------------\n');
  }
}

main().catch(console.error);

