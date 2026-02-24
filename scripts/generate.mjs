#!/usr/bin/env node
/**
 * Apply renames to JavaScript code using Babel AST
 * Usage:
 *   node generate.mjs <input.js> <renames.json> [output.js]
 *   node generate.mjs --stdin
 *
 * renames.json format: {"oldName": "newName", "a:10": "tempValue"}
 * - "oldName" -> rename all occurrences in scope
 * - "a:10" -> rename only at line 10
 */

import { parse } from '@babel/parser';
import traverse from '@babel/traverse';
import generate from '@babel/generator';
import fs from 'fs';

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

function buildRenameMaps(renamesData) {
  const globalRenames = {}; // {"a": "newName"}
  const lineSpecificRenames = {}; // {lineNumber: {"a": "newName"}}

  for (const [key, newName] of Object.entries(renamesData || {})) {
    if (key.includes(':')) {
      const [name, lineStr] = key.split(':');
      const line = parseInt(lineStr, 10);
      if (!Number.isInteger(line)) {
        continue;
      }
      if (!lineSpecificRenames[line]) {
        lineSpecificRenames[line] = {};
      }
      lineSpecificRenames[line][name] = newName;
    } else {
      globalRenames[key] = newName;
    }
  }

  return { globalRenames, lineSpecificRenames };
}

async function maybeBeautify(code, enabled) {
  if (!enabled) {
    return code;
  }

  try {
    const prettier = await import('prettier');
    return await prettier.format(code, {
      parser: 'babel',
      printWidth: 120,
    });
  } catch {
    return code;
  }
}

async function transformCode(code, renamesData, options = {}) {
  const { globalRenames, lineSpecificRenames } = buildRenameMaps(renamesData);
  const retainLines = options.retainLines !== false;

  let ast;
  try {
    ast = parse(code, parseOptions);
  } catch (e) {
    throw new Error(`Parse error: ${e.message}`);
  }

  // Track renamed binding paths to avoid renaming references to different bindings.
  const renamedBindings = new Map(); // scopePath -> {oldName: newName}

  // Traverse and rename.
  traverse.default(ast, {
    BindingIdentifier(path) {
      const name = path.node.name;
      const line = path.node.loc ? path.node.loc.start.line : 0;

      if (lineSpecificRenames[line] && lineSpecificRenames[line][name]) {
        const newName = lineSpecificRenames[line][name];
        path.node.name = newName;

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

      if (globalRenames[name]) {
        const newName = globalRenames[name];
        path.node.name = newName;

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

    ReferencedIdentifier(path) {
      const name = path.node.name;
      const binding = path.scope.getBinding(name);
      if (binding) {
        const scopeId = binding.scope.uid;
        const scopeRenames = renamedBindings.get(scopeId);

        if (scopeRenames && scopeRenames[name]) {
          path.node.name = scopeRenames[name];
          return;
        }
      }

      // Fallback for undeclared identifiers (global/built-ins).
      if (globalRenames[name] && !binding) {
        path.node.name = globalRenames[name];
      }
    },
  });

  const generated = generate.default(
    ast,
    {
      comments: true,
      compact: false,
      retainLines,
      concise: false,
    },
    code,
  );

  return maybeBeautify(generated.code, options.beautify === true);
}

function parseStdinPayload(rawText) {
  let payload;
  try {
    payload = JSON.parse(rawText);
  } catch (e) {
    throw new Error(`Invalid stdin JSON payload: ${e.message}`);
  }

  if (!payload || typeof payload !== 'object') {
    throw new Error('Invalid stdin payload: expected object');
  }

  return {
    code: typeof payload.code === 'string' ? payload.code : '',
    renames: payload.renames && typeof payload.renames === 'object' ? payload.renames : {},
    outputPath: typeof payload.outputPath === 'string' && payload.outputPath ? payload.outputPath : null,
    options: payload.options && typeof payload.options === 'object' ? payload.options : {},
  };
}

async function main() {
  const args = process.argv.slice(2);
  const stdinMode = args[0] === '--stdin';
  let code = '';
  let renamesData = {};
  let outputPath = null;
  let options = {};

  if (stdinMode) {
    const rawPayload = fs.readFileSync(0, 'utf-8');
    const payload = parseStdinPayload(rawPayload);
    code = payload.code;
    renamesData = payload.renames;
    outputPath = payload.outputPath;
    options = payload.options;
  } else {
    if (args.length < 2) {
      console.error('Usage: node generate.mjs <input.js> <renames.json> [output.js]');
      process.exit(1);
    }

    const inputPath = args[0];
    const renamesPath = args[1];
    outputPath = args[2] || null;
    code = fs.readFileSync(inputPath, 'utf-8');
    renamesData = JSON.parse(fs.readFileSync(renamesPath, 'utf-8'));
  }

  const transformed = await transformCode(code, renamesData, options);
  if (outputPath) {
    fs.writeFileSync(outputPath, transformed, 'utf-8');
    if (!stdinMode) {
      console.log(`Generated: ${outputPath}`);
    }
    return;
  }

  process.stdout.write(transformed);
}

main().catch((e) => {
  console.error(e.message || String(e));
  process.exit(1);
});
