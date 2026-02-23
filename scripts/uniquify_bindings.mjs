#!/usr/bin/env node
/**
 * Ensure all binding identifiers are globally unique across the file.
 * Usage: node uniquify_bindings.mjs <input.js> [output.js]
 */

import { parse } from '@babel/parser';
import traverse from '@babel/traverse';
import generate from '@babel/generator';
import fs from 'fs';

const args = process.argv.slice(2);
if (args.length < 1) {
  console.error('Usage: node uniquify_bindings.mjs <input.js> [output.js]');
  process.exit(1);
}

const inputPath = args[0];
const outputPath = args[1] || null;
const code = fs.readFileSync(inputPath, 'utf-8');

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
    'recordAndTuple',
    'v8intrinsic',
    'explicitResourceManagement',
    'decoratorAutoAccessors',
  ],
  errorRecovery: true,
  tokens: false,
  comment: true,
};

let ast;
try {
  ast = parse(code, parseOptions);
} catch (e) {
  console.error(`Parse error: ${e.message}`);
  process.exit(1);
}

const seenBindings = new Set();
const groups = new Map();
const usedNames = new Set();

function bindingKey(binding, name) {
  if (!binding?.identifier) {
    return null;
  }
  const id = binding.identifier;
  if (typeof id.start === 'number' && typeof id.end === 'number') {
    return `${name}:${id.start}:${id.end}`;
  }
  if (id.loc) {
    return `${name}:${id.loc.start.line}:${id.loc.start.column}`;
  }
  return null;
}

function pushBinding(name, binding) {
  if (!groups.has(name)) {
    groups.set(name, []);
  }
  groups.get(name).push(binding);
}

function getBindingStart(binding) {
  if (!binding?.identifier) {
    return Number.MAX_SAFE_INTEGER;
  }
  if (typeof binding.identifier.start === 'number') {
    return binding.identifier.start;
  }
  if (binding.identifier.loc) {
    return binding.identifier.loc.start.line * 100000 + binding.identifier.loc.start.column;
  }
  return Number.MAX_SAFE_INTEGER;
}

function makeUniqueName(baseName, index) {
  let suffix = index;
  let candidate = `${baseName}__u${suffix}`;
  while (usedNames.has(candidate)) {
    suffix += 1;
    candidate = `${baseName}__u${suffix}`;
  }
  usedNames.add(candidate);
  return candidate;
}

traverse.default(ast, {
  Scope(path) {
    for (const [name, binding] of Object.entries(path.scope.bindings)) {
      if (!binding || binding.kind === 'module') {
        continue;
      }

      usedNames.add(name);
      const key = bindingKey(binding, name);
      if (!key || seenBindings.has(key)) {
        continue;
      }

      seenBindings.add(key);
      pushBinding(name, binding);
    }
  },
});

for (const [name, bindings] of groups.entries()) {
  if (bindings.length <= 1) {
    continue;
  }

  bindings.sort((a, b) => getBindingStart(a) - getBindingStart(b));

  let index = 1;
  for (const binding of bindings) {
    const uniqueName = makeUniqueName(name, index);
    binding.scope.rename(name, uniqueName);
    index += 1;
  }
}

const output = generate.default(ast, {
  comments: true,
  compact: false,
  retainLines: false,
  concise: false,
}, code);

if (outputPath) {
  fs.writeFileSync(outputPath, output.code, 'utf-8');
  console.log(`Generated: ${outputPath}`);
} else {
  console.log(output.code);
}
