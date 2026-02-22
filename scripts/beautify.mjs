#!/usr/bin/env node
/**
 * Beautify JavaScript code using js-beautify
 * Usage: node beautify.mjs <input_file> <output_file>
 */

import pkg from 'js-beautify';
const { js } = pkg;
import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);

if (args.length < 2) {
  console.error('Usage: node beautify.mjs <input_file> <output_file>');
  process.exit(1);
}

const inputPath = args[0];
const outputPath = args[1];

// Check if input file exists
if (!fs.existsSync(inputPath)) {
  console.error(`Error: Input file does not exist: ${inputPath}`);
  process.exit(1);
}

// Ensure output directory exists
const outputDir = path.dirname(outputPath);
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

const original = fs.readFileSync(path.resolve(inputPath), 'utf-8');

const beautified = js(original, {
  indent_size: 2,
  space_in_empty_paren: true,
  max_preserve_newlines: 2,
  preserve_newlines: true,
  jslint_happy: false,
  keep_array_indentation: false,
  keep_function_indentation: false,
  break_chained_methods: true,
  brace_style: 'collapse',
  space_before_conditional: true,
  unescape_strings: false,
  wrap_line_length: 120,  // Limit line length
  wrap_attributes: 'force-expand-multiline',
  e4x: false,
});

fs.writeFileSync(path.resolve(outputPath), beautified);

console.log(`Beautified: ${inputPath} -> ${outputPath}`);
