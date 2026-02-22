#!/usr/bin/env node
/**
 * Apply renames to JavaScript code using Babel AST
 * Usage: node generate.mjs <input.js> <renames.json> [output.js]
 *
 * renames.json format: {"oldName": "newName", "a:10": "tempValue"}
 * - "oldName" -> rename all occurrences in scope
 * - "a:10" -> rename only at line 10
 */

import { parse } from '@babel/parser';
import traverse from '@babel/traverse';
import generate from '@babel/generator';
import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);

if (args.length < 2) {
  console.error('Usage: node generate.mjs <input.js> <renames.json> [output.js]');
  process.exit(1);
}

const inputPath = args[0];
const renamesPath = args[1];
const outputPath = args[2] || null;

// Read input files
const code = fs.readFileSync(inputPath, 'utf-8');
const renamesData = JSON.parse(fs.readFileSync(renamesPath, 'utf-8'));

// Parse renames
const globalRenames = {};  // {"a": "newName"}
const lineSpecificRenames = {};  // {lineNumber: {"a": "newName"}}

for (const [key, newName] of Object.entries(renamesData)) {
  if (key.includes(':')) {
    const [name, lineStr] = key.split(':');
    const line = parseInt(lineStr, 10);
    if (!lineSpecificRenames[line]) {
      lineSpecificRenames[line] = {};
    }
    lineSpecificRenames[line][name] = newName;
  } else {
    globalRenames[key] = newName;
  }
}

// Parse options
const parseOptions = {
  sourceType: 'unambiguous',
  plugins: [
    'jsx',
    'typescript',
    'decorators',
    'classProperties',
    'classPrivateProperties',
    'classPrivateMethods',
    'exportDefaultFrom',
    'exportNamespaceFrom',
    'dynamicImport',
    'nullishCoalescingOperator',
    'optionalChaining',
    'numericSeparator',
    'bigInt',
    'objectRestSpread',
    'optionalCatchBinding',
    'asyncGenerators',
    'functionBind',
    'functionSent',
    'logicalAssignment',
    'importMeta',
    'topLevelAwait',
    'privatePropertyInObject',
    'moduleStringNames',
  ],
  errorRecovery: true,
  tokens: false,
  comment: true,
};

// Parse
let ast;
try {
  ast = parse(code, parseOptions);
} catch (e) {
  console.error(`Parse error: ${e.message}`);
  process.exit(1);
}

// Track renamed binding paths to avoid renaming references to different bindings
const renamedBindings = new Map(); // scopePath -> {oldName: newName}

// Traverse and rename
traverse.default(ast, {
  // Rename binding declarations
  BindingIdentifier(path) {
    const name = path.node.name;
    const line = path.node.loc ? path.node.loc.start.line : 0;

    // Check line-specific rename first
    if (lineSpecificRenames[line] && lineSpecificRenames[line][name]) {
      const newName = lineSpecificRenames[line][name];
      path.node.name = newName;

      // Record this binding was renamed
      const binding = path.scope.getBinding(name);
      if (binding) {
        const scopeId = binding.scope.uid;
        if (!renamedBindings.has(scopeId)) {
          renamedBindings.set(scopeId, {});
        }
        renamedBindings.get(scopeId)[name] = newName;
      }
      return;
    }

    // Check global rename
    if (globalRenames[name]) {
      const newName = globalRenames[name];
      path.node.name = newName;

      // Record this binding was renamed
      const binding = path.scope.getBinding(name);
      if (binding) {
        const scopeId = binding.scope.uid;
        if (!renamedBindings.has(scopeId)) {
          renamedBindings.set(scopeId, {});
        }
        renamedBindings.get(scopeId)[name] = newName;
      }
    }
  },

  // Rename references to bindings that were renamed
  ReferencedIdentifier(path) {
    const name = path.node.name;

    // Check if this references a binding that was renamed
    const binding = path.scope.getBinding(name);
    if (binding) {
      const scopeId = binding.scope.uid;
      const scopeRenames = renamedBindings.get(scopeId);

      if (scopeRenames && scopeRenames[name]) {
        path.node.name = scopeRenames[name];
        return;
      }
    }

    // Fallback: check global renames for undeclared identifiers
    // (e.g., global variables, built-ins used as references)
    if (globalRenames[name] && !binding) {
      path.node.name = globalRenames[name];
    }
  },
});

// Generate code
const output = generate.default(ast, {
  comments: true,
  compact: false,
  retainLines: false,
  concise: false,
}, code);

// Output
if (outputPath) {
  fs.writeFileSync(outputPath, output.code, 'utf-8');
  console.log(`Generated: ${outputPath}`);
} else {
  console.log(output.code);
}
