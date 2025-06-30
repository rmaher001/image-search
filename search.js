"use strict";
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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var transformers_1 = require("@xenova/transformers");
var fs = require("fs/promises");
var path = require("path");
// --- CONFIGURATION ---
var SCRIPT_VERSION = '3.0.1';
var MODEL_NAME = 'Xenova/clip-vit-base-patch32';
var DB_PATH = './db.json';
var IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
/**
 * Manages the loading of the models, processor, and tokenizer to ensure they are
 * loaded only once using a singleton pattern. Now uses the correct, specific
 * model classes for vision and text.
 */
var CLIPSingleton = /** @class */ (function () {
    function CLIPSingleton() {
    }
    CLIPSingleton.getInstance = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, _b, _c, _d;
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0:
                        if (!(this.tokenizer === null)) return [3 /*break*/, 2];
                        console.log('Loading tokenizer...');
                        _a = this;
                        return [4 /*yield*/, transformers_1.AutoTokenizer.from_pretrained(MODEL_NAME)];
                    case 1:
                        _a.tokenizer = _e.sent();
                        console.log('Tokenizer loaded.');
                        _e.label = 2;
                    case 2:
                        if (!(this.processor === null)) return [3 /*break*/, 4];
                        console.log('Loading image processor...');
                        _b = this;
                        return [4 /*yield*/, transformers_1.AutoProcessor.from_pretrained(MODEL_NAME)];
                    case 3:
                        _b.processor = _e.sent();
                        console.log('Image processor loaded.');
                        _e.label = 4;
                    case 4:
                        if (!(this.visionModel === null)) return [3 /*break*/, 6];
                        console.log('Loading CLIP Vision model... This may take a moment.');
                        _c = this;
                        return [4 /*yield*/, transformers_1.CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true })];
                    case 5:
                        _c.visionModel = _e.sent();
                        console.log('Vision model loaded.');
                        _e.label = 6;
                    case 6:
                        if (!(this.textModel === null)) return [3 /*break*/, 8];
                        console.log('Loading CLIP Text model... This may take a moment.');
                        _d = this;
                        return [4 /*yield*/, transformers_1.CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, { quantized: true })];
                    case 7:
                        _d.textModel = _e.sent();
                        console.log('Text model loaded.');
                        _e.label = 8;
                    case 8: return [2 /*return*/, {
                            tokenizer: this.tokenizer,
                            processor: this.processor,
                            visionModel: this.visionModel,
                            textModel: this.textModel,
                        }];
                }
            });
        });
    };
    CLIPSingleton.tokenizer = null;
    CLIPSingleton.processor = null;
    CLIPSingleton.visionModel = null;
    CLIPSingleton.textModel = null;
    return CLIPSingleton;
}());
/**
 * Calculates the cosine similarity between two vectors.
 */
function cosineSimilarity(v1, v2) {
    var dotProduct = v1.reduce(function (sum, a, i) { return sum + a * v2[i]; }, 0);
    var magnitude1 = Math.sqrt(v1.reduce(function (sum, a) { return sum + a * a; }, 0));
    var magnitude2 = Math.sqrt(v2.reduce(function (sum, a) { return sum + a * a; }, 0));
    if (magnitude1 === 0 || magnitude2 === 0)
        return 0;
    return dotProduct / (magnitude1 * magnitude2);
}
/**
 * Generates an embedding for a given image file.
 */
function embedImage(visionModel, processor, filePath) {
    return __awaiter(this, void 0, void 0, function () {
        var image, image_inputs, image_embeds;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, transformers_1.RawImage.fromURL(filePath)];
                case 1:
                    image = _a.sent();
                    return [4 /*yield*/, processor(image)];
                case 2:
                    image_inputs = _a.sent();
                    return [4 /*yield*/, visionModel(image_inputs)];
                case 3:
                    image_embeds = (_a.sent()).image_embeds;
                    return [2 /*return*/, Array.from(image_embeds.data)];
            }
        });
    });
}
/**
 * Generates an embedding for a given text query.
 */
function embedText(textModel, tokenizer, text) {
    return __awaiter(this, void 0, void 0, function () {
        var text_inputs, text_embeds;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    text_inputs = tokenizer(text, { padding: true, truncation: true });
                    return [4 /*yield*/, textModel(text_inputs)];
                case 1:
                    text_embeds = (_a.sent()).text_embeds;
                    return [2 /*return*/, Array.from(text_embeds.data)];
            }
        });
    });
}
/**
 * Recursively finds all image files in a directory and its subdirectories.
 */
function findImageFilesRecursive(dir) {
    return __awaiter(this, void 0, void 0, function () {
        var imageFiles, files, _i, files_1, file, fullPath, _a, _b, error_1;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    imageFiles = [];
                    _c.label = 1;
                case 1:
                    _c.trys.push([1, 8, , 9]);
                    return [4 /*yield*/, fs.readdir(dir, { withFileTypes: true })];
                case 2:
                    files = _c.sent();
                    _i = 0, files_1 = files;
                    _c.label = 3;
                case 3:
                    if (!(_i < files_1.length)) return [3 /*break*/, 7];
                    file = files_1[_i];
                    fullPath = path.join(dir, file.name);
                    if (!file.isDirectory()) return [3 /*break*/, 5];
                    _b = (_a = imageFiles).concat;
                    return [4 /*yield*/, findImageFilesRecursive(fullPath)];
                case 4:
                    imageFiles = _b.apply(_a, [_c.sent()]);
                    return [3 /*break*/, 6];
                case 5:
                    if (IMAGE_EXTENSIONS.includes(path.extname(file.name).toLowerCase())) {
                        imageFiles.push(fullPath);
                    }
                    _c.label = 6;
                case 6:
                    _i++;
                    return [3 /*break*/, 3];
                case 7: return [3 /*break*/, 9];
                case 8:
                    error_1 = _c.sent();
                    console.warn("Could not read directory ".concat(dir, ": ").concat(error_1.message));
                    return [3 /*break*/, 9];
                case 9: return [2 /*return*/, imageFiles];
            }
        });
    });
}
/**
 * INDEXING FUNCTION
 */
function indexImages(rootDir, limit) {
    return __awaiter(this, void 0, void 0, function () {
        var _a, visionModel, processor, db, imageFiles, count, _i, imageFiles_1, filePath, embedding, e_1, errorMessage, error_2;
        var _b;
        return __generator(this, function (_c) {
            switch (_c.label) {
                case 0:
                    console.log('🚀 Starting image indexing (recursive search)...');
                    if (limit) {
                        console.log("\u26A0\uFE0F  Processing limit set to ".concat(limit, " images."));
                    }
                    return [4 /*yield*/, CLIPSingleton.getInstance()];
                case 1:
                    _a = _c.sent(), visionModel = _a.visionModel, processor = _a.processor;
                    db = [];
                    _c.label = 2;
                case 2:
                    _c.trys.push([2, 11, , 12]);
                    return [4 /*yield*/, findImageFilesRecursive(rootDir)];
                case 3:
                    imageFiles = _c.sent();
                    if (imageFiles.length === 0) {
                        console.log("\uD83D\uDFE1 No images found in \"".concat(rootDir, "\" or its subdirectories."));
                        return [2 /*return*/];
                    }
                    console.log("Found ".concat(imageFiles.length, " total images to process."));
                    count = 0;
                    _i = 0, imageFiles_1 = imageFiles;
                    _c.label = 4;
                case 4:
                    if (!(_i < imageFiles_1.length)) return [3 /*break*/, 9];
                    filePath = imageFiles_1[_i];
                    if (limit && count >= limit) {
                        console.log("Reached processing limit of ".concat(limit, ". Stopping."));
                        return [3 /*break*/, 9];
                    }
                    count++;
                    console.log("  [".concat(count, "/").concat(imageFiles.length, "] Processing ").concat(filePath, "..."));
                    _c.label = 5;
                case 5:
                    _c.trys.push([5, 7, , 8]);
                    return [4 /*yield*/, embedImage(visionModel, processor, filePath)];
                case 6:
                    embedding = _c.sent();
                    db.push({ filePath: filePath, embedding: embedding });
                    return [3 /*break*/, 8];
                case 7:
                    e_1 = _c.sent();
                    errorMessage = (_b = e_1 === null || e_1 === void 0 ? void 0 : e_1.message) !== null && _b !== void 0 ? _b : String(e_1);
                    console.warn("  Skipping file ".concat(filePath, " due to an error: ").concat(errorMessage));
                    return [3 /*break*/, 8];
                case 8:
                    _i++;
                    return [3 /*break*/, 4];
                case 9: return [4 /*yield*/, fs.writeFile(DB_PATH, JSON.stringify(db, null, 2))];
                case 10:
                    _c.sent();
                    console.log("\u2705 Indexing complete! Database with ".concat(db.length, " entries saved to ").concat(DB_PATH));
                    return [3 /*break*/, 12];
                case 11:
                    error_2 = _c.sent();
                    console.error("\u274C Error during indexing: ".concat(error_2.message));
                    return [3 /*break*/, 12];
                case 12: return [2 /*return*/];
            }
        });
    });
}
/**
 * SEARCH FUNCTION
 */
function search(query) {
    return __awaiter(this, void 0, void 0, function () {
        var db, dbFile, error_3, _a, textModel, tokenizer, queryEmbedding, bestMatch, highestSimilarity, _i, db_1, entry, similarity;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    console.log("\uD83D\uDD0E Searching for: \"".concat(query, "\""));
                    _b.label = 1;
                case 1:
                    _b.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, fs.readFile(DB_PATH, 'utf-8')];
                case 2:
                    dbFile = _b.sent();
                    db = JSON.parse(dbFile);
                    return [3 /*break*/, 4];
                case 3:
                    error_3 = _b.sent();
                    console.error("\u274C Database file not found at ".concat(DB_PATH, "."));
                    console.error('Please run the "index" command first.');
                    return [2 /*return*/];
                case 4:
                    if (db.length === 0) {
                        console.error('❌ The database is empty.');
                        return [2 /*return*/];
                    }
                    return [4 /*yield*/, CLIPSingleton.getInstance()];
                case 5:
                    _a = _b.sent(), textModel = _a.textModel, tokenizer = _a.tokenizer;
                    return [4 /*yield*/, embedText(textModel, tokenizer, query)];
                case 6:
                    queryEmbedding = _b.sent();
                    bestMatch = null;
                    highestSimilarity = -Infinity;
                    for (_i = 0, db_1 = db; _i < db_1.length; _i++) {
                        entry = db_1[_i];
                        similarity = cosineSimilarity(queryEmbedding, entry.embedding);
                        if (similarity > highestSimilarity) {
                            highestSimilarity = similarity;
                            bestMatch = entry;
                        }
                    }
                    if (bestMatch) {
                        console.log('\n✨ Best match found! ✨');
                        console.log("   File: ".concat(bestMatch.filePath));
                        console.log("   Similarity Score: ".concat((highestSimilarity * 100).toFixed(2), "%"));
                    }
                    else {
                        console.log('Could not find a suitable match.');
                    }
                    return [2 /*return*/];
            }
        });
    });
}
/**
 * MAIN FUNCTION
 */
function main() {
    return __awaiter(this, void 0, void 0, function () {
        var args, command, dirOrQuery, limitArg, limit, query;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    console.log("\n--- Image Search CLI v".concat(SCRIPT_VERSION, " ---"));
                    args = process.argv.slice(2);
                    command = args[0];
                    dirOrQuery = args[1];
                    limitArg = args[2];
                    if (!(command === 'index' && dirOrQuery)) return [3 /*break*/, 2];
                    limit = limitArg ? parseInt(limitArg, 10) : undefined;
                    if (limitArg && isNaN(limit)) {
                        console.error('Error: The limit argument must be a number.');
                        return [2 /*return*/];
                    }
                    return [4 /*yield*/, indexImages(dirOrQuery, limit)];
                case 1:
                    _a.sent();
                    return [3 /*break*/, 5];
                case 2:
                    if (!(command === 'search' && dirOrQuery)) return [3 /*break*/, 4];
                    query = args.slice(1).join(' ');
                    return [4 /*yield*/, search(query)];
                case 3:
                    _a.sent();
                    return [3 /*break*/, 5];
                case 4:
                    console.log('-------------------------------------------');
                    console.log(" TypeScript Image Search CLI (v".concat(SCRIPT_VERSION, ")"));
                    console.log('-------------------------------------------\n');
                    console.log('Usage:');
                    console.log('  To index a folder of images:');
                    console.log('    ts-node search.ts index <path_to_folder> [limit]\n');
                    console.log('  To search for an image with a text query:');
                    console.log('    ts-node search.ts search "your search query here"');
                    console.log('-------------------------------------------\n');
                    _a.label = 5;
                case 5: return [2 /*return*/];
            }
        });
    });
}
main().catch(console.error);
