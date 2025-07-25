# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a TypeScript-based image search application using OpenAI's CLIP model for semantic image search. It uses the `@xenova/transformers` library to run a quantized version of CLIP directly in Node.js.

## Architecture

The application consists of two main components:

1. **Indexing**: Recursively scans directories for images, generates CLIP embeddings, and stores them in `db.json`
2. **Search**: Takes text queries, generates text embeddings, and finds most similar images using cosine similarity

Key architectural decisions:
- Uses singleton pattern for CLIP model management (CLIPSingleton class) to ensure models are loaded only once
- Embeddings are persisted in a simple JSON database (`db.json`)
- Supports recursive directory scanning for images
- Implements semantic search with configurable similarity threshold (0.28) and top-N results

## Development Commands

```bash
# Install dependencies
npm install

# Run TypeScript directly (for development)
npx ts-node search.ts index ./path/to/images [limit]
npx ts-node search.ts search "your search query" [top_n]

# Compile TypeScript to JavaScript
npx tsc

# Run compiled JavaScript
node dist/search.js index ./path/to/images [limit]
node dist/search.js search "your search query" [top_n]
```

## Configuration

Key configuration constants in `search.ts`:
- `MODEL_NAME`: 'Xenova/clip-vit-base-patch32'
- `DB_PATH`: './db.json'
- `IMAGE_EXTENSIONS`: ['.png', '.jpg', '.jpeg', '.webp']
- `SIMILARITY_THRESHOLD`: 0.28 (minimum score to show results)

## Dependencies

- `@xenova/transformers`: CLIP model implementation
- TypeScript dev dependencies: `typescript`, `ts-node`, `@types/node`

## Important Guidance

- **Never add co-authored by Claude or any similar language to the comments or code base**