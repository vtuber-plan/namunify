#!/usr/bin/env node
/**
 * JavaScript AST Parser using Babel
 * Usage: node js_parser.mjs <input.js> [options]
 * Output: JSON to stdout
 */

import { parse } from '@babel/parser';
import traverse from '@babel/traverse';
import fs from 'fs';

const inputPath = process.argv[2];
if (!inputPath) {
  console.error('Usage: node js_parser.mjs <input.js>');
  process.exit(1);
}

const code = fs.readFileSync(inputPath, 'utf-8');
const lines = code.split('\n');

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
    'recordAndTuple',
    'v8intrinsic',
    'explicitResourceManagement',
    'decoratorAutoAccessors',
  ],
  errorRecovery: true,
  tokens: false,
  comment: false,
};

let ast;
try {
  ast = parse(code, parseOptions);
} catch (e) {
  console.error(`Parse error: ${e.message}`);
  process.exit(1);
}

// Result structure
const result = {
  sourceCode: code,
  lines: lines,
  scopes: {},
  bindings: [],
};

// Scope tracking
let scopeId = 0;
const scopeStack = [];
const allScopes = {};

function createScope(type, startLine, startCol, endLine, endCol, parentId = null) {
  const id = `scope_${scopeId++}`;
  const scope = {
    scopeId: id,
    type: type,
    range: {
      start: { line: startLine, column: startCol },
      end: { line: endLine, column: endCol },
    },
    bindings: [],
    parentId: parentId,
    children: [],
  };
  allScopes[id] = scope;
  return id;
}

// Get position from node
function getPosition(node) {
  return {
    line: node.loc ? node.loc.start.line - 1 : 0, // Convert to 0-indexed
    column: node.loc ? node.loc.start.column : 0,
  };
}

function getEndPosition(node) {
  return {
    line: node.loc ? node.loc.end.line - 1 : 0,
    column: node.loc ? node.loc.end.column : 0,
  };
}

// Check if name looks obfuscated
function isObfuscatedName(name) {
  if (!name) return false;

  // Preserve common meaningful names
  const preserved = new Set([
    'i', 'j', 'k', 'x', 'y', 'z', 'id', 'db', 'io', 'ui', 'os',
    'fs', 'ls', 'rm', 'cp', 'up', 'on', 'in', 'to', 'fn', 'cb',
    'ev', 'el', 'tx', 'rx', 'req', 'res', 'err', 'ctx', 'app',
    'set', 'get', 'map', 'key', 'val', 'arr', 'obj', 'str', 'num',
  ]);

  if (preserved.has(name.toLowerCase())) return false;

  // Single letter (except preserved)
  if (name.length === 1) return true;

  // Two chars without clear meaning
  if (name.length === 2) {
    const common2 = ['id', 'io', 'ui', 'db', 'fn', 'cb', 'ev', 'el', 'tx', 'rx', 'if', 'do', 'of'];
    if (common2.includes(name.toLowerCase())) return false;
    return true;
  }

  // Letter + number pattern
  if (/^[a-zA-Z]\d+$/.test(name)) return true;

  // Short random letter combos
  if (name.length <= 3 && /^[a-zA-Z]+$/.test(name)) {
    const meaningful = ['set', 'get', 'map', 'key', 'val', 'arr', 'obj', 'str', 'num', 'err', 'log', 'run', 'add', 'sub', 'mul', 'div', 'new', 'old', 'src', 'dst'];
    if (meaningful.includes(name.toLowerCase())) return false;
    if (new Set(name.toLowerCase()).size <= 2) return true;
  }

  // Underscore with short name
  if (name.includes('_') && name.length <= 5) return true;

  return false;
}

// Create program scope
const programScopeId = createScope(
  'program',
  0, 0,
  lines.length - 1, lines[lines.length - 1]?.length || 0
);
scopeStack.push(programScopeId);

// Traverse AST
traverse.default(ast, {
  // Function declarations create new scopes
  FunctionDeclaration(path) {
    const node = path.node;
    const start = getPosition(node);
    const end = getEndPosition(node);
    const parentId = scopeStack[scopeStack.length - 1];

    const scopeId = createScope('function', start.line, start.column, end.line, end.column, parentId);
    allScopes[parentId].children.push(scopeId);

    // Function name is in parent scope
    if (node.id && node.id.name) {
      allScopes[parentId].bindings.push({
        name: node.id.name,
        line: getPosition(node.id).line,
        column: getPosition(node.id).column,
        type: 'function',
        isObfuscated: isObfuscatedName(node.id.name),
      });
    }

    scopeStack.push(scopeId);

    // Parameters are in function scope
    node.params.forEach(param => {
      if (param.type === 'Identifier' && param.name) {
        allScopes[scopeId].bindings.push({
          name: param.name,
          line: getPosition(param).line,
          column: getPosition(param).column,
          type: 'parameter',
          isObfuscated: isObfuscatedName(param.name),
        });
      } else if (param.type === 'AssignmentPattern' && param.left?.type === 'Identifier') {
        allScopes[scopeId].bindings.push({
          name: param.left.name,
          line: getPosition(param.left).line,
          column: getPosition(param.left).column,
          type: 'parameter',
          isObfuscated: isObfuscatedName(param.left.name),
        });
      } else if (param.type === 'RestElement' && param.argument?.type === 'Identifier') {
        allScopes[scopeId].bindings.push({
          name: param.argument.name,
          line: getPosition(param.argument).line,
          column: getPosition(param.argument).column,
          type: 'parameter',
          isObfuscated: isObfuscatedName(param.argument.name),
        });
      }
    });
  },

  FunctionExpression(path) {
    const node = path.node;
    const start = getPosition(node);
    const end = getEndPosition(node);
    const parentId = scopeStack[scopeStack.length - 1];

    const scopeId = createScope('function', start.line, start.column, end.line, end.column, parentId);
    allScopes[parentId].children.push(scopeId);

    // Named function expression
    if (node.id && node.id.name) {
      allScopes[scopeId].bindings.push({
        name: node.id.name,
        line: getPosition(node.id).line,
        column: getPosition(node.id).column,
        type: 'function',
        isObfuscated: isObfuscatedName(node.id.name),
      });
    }

    scopeStack.push(scopeId);

    node.params.forEach(param => {
      if (param.type === 'Identifier' && param.name) {
        allScopes[scopeId].bindings.push({
          name: param.name,
          line: getPosition(param).line,
          column: getPosition(param).column,
          type: 'parameter',
          isObfuscated: isObfuscatedName(param.name),
        });
      }
    });
  },

  ArrowFunctionExpression(path) {
    const node = path.node;
    const start = getPosition(node);
    const end = getEndPosition(node);
    const parentId = scopeStack[scopeStack.length - 1];

    const scopeId = createScope('arrow', start.line, start.column, end.line, end.column, parentId);
    allScopes[parentId].children.push(scopeId);

    scopeStack.push(scopeId);

    node.params.forEach(param => {
      if (param.type === 'Identifier' && param.name) {
        allScopes[scopeId].bindings.push({
          name: param.name,
          line: getPosition(param).line,
          column: getPosition(param).column,
          type: 'parameter',
          isObfuscated: isObfuscatedName(param.name),
        });
      }
    });
  },

  ClassDeclaration(path) {
    const node = path.node;
    const start = getPosition(node);
    const end = getEndPosition(node);
    const parentId = scopeStack[scopeStack.length - 1];

    const scopeId = createScope('class', start.line, start.column, end.line, end.column, parentId);
    allScopes[parentId].children.push(scopeId);

    if (node.id && node.id.name) {
      allScopes[parentId].bindings.push({
        name: node.id.name,
        line: getPosition(node.id).line,
        column: getPosition(node.id).column,
        type: 'class',
        isObfuscated: isObfuscatedName(node.id.name),
      });
    }

    scopeStack.push(scopeId);
  },

  ClassExpression(path) {
    const node = path.node;
    const start = getPosition(node);
    const end = getEndPosition(node);
    const parentId = scopeStack[scopeStack.length - 1];

    const scopeId = createScope('class', start.line, start.column, end.line, end.column, parentId);
    allScopes[parentId].children.push(scopeId);

    if (node.id && node.id.name) {
      allScopes[scopeId].bindings.push({
        name: node.id.name,
        line: getPosition(node.id).line,
        column: getPosition(node.id).column,
        type: 'class',
        isObfuscated: isObfuscatedName(node.id.name),
      });
    }

    scopeStack.push(scopeId);
  },

  // Variable declarations
  VariableDeclarator(path) {
    const node = path.node;
    const currentScopeId = scopeStack[scopeStack.length - 1];
    const currentScope = allScopes[currentScopeId];

    if (node.id.type === 'Identifier' && node.id.name) {
      currentScope.bindings.push({
        name: node.id.name,
        line: getPosition(node.id).line,
        column: getPosition(node.id).column,
        type: 'variable',
        isObfuscated: isObfuscatedName(node.id.name),
      });
    } else if (node.id.type === 'ObjectPattern') {
      // Destructuring: const { a, b } = obj
      node.id.properties.forEach(prop => {
        if (prop.type === 'RestElement' && prop.argument?.name) {
          currentScope.bindings.push({
            name: prop.argument.name,
            line: getPosition(prop.argument).line,
            column: getPosition(prop.argument).column,
            type: 'variable',
            isObfuscated: isObfuscatedName(prop.argument.name),
          });
        } else if (prop.value?.type === 'Identifier') {
          currentScope.bindings.push({
            name: prop.value.name,
            line: getPosition(prop.value).line,
            column: getPosition(prop.value).column,
            type: 'variable',
            isObfuscated: isObfuscatedName(prop.value.name),
          });
        }
      });
    } else if (node.id.type === 'ArrayPattern') {
      // Destructuring: const [a, b] = arr
      node.id.elements.forEach(elem => {
        if (elem?.type === 'Identifier') {
          currentScope.bindings.push({
            name: elem.name,
            line: getPosition(elem).line,
            column: getPosition(elem).column,
            type: 'variable',
            isObfuscated: isObfuscatedName(elem.name),
          });
        }
      });
    }
  },

  // For statement creates block scope
  ForStatement(path) {
    const node = path.node;
    if (node.body?.type === 'BlockStatement') {
      const start = getPosition(node.body);
      const end = getEndPosition(node.body);
      const parentId = scopeStack[scopeStack.length - 1];

      const scopeId = createScope('block', start.line, start.column, end.line, end.column, parentId);
      allScopes[parentId].children.push(scopeId);
      scopeStack.push(scopeId);
    }
  },

  ForInStatement(path) {
    const node = path.node;
    if (node.body?.type === 'BlockStatement') {
      const start = getPosition(node.body);
      const end = getEndPosition(node.body);
      const parentId = scopeStack[scopeStack.length - 1];

      const scopeId = createScope('block', start.line, start.column, end.line, end.column, parentId);
      allScopes[parentId].children.push(scopeId);
      scopeStack.push(scopeId);
    }
  },

  ForOfStatement(path) {
    const node = path.node;
    if (node.body?.type === 'BlockStatement') {
      const start = getPosition(node.body);
      const end = getEndPosition(node.body);
      const parentId = scopeStack[scopeStack.length - 1];

      const scopeId = createScope('block', start.line, start.column, end.line, end.column, parentId);
      allScopes[parentId].children.push(scopeId);
      scopeStack.push(scopeId);
    }
  },

  CatchClause(path) {
    const node = path.node;
    const start = getPosition(node);
    const end = getEndPosition(node);
    const parentId = scopeStack[scopeStack.length - 1];

    const scopeId = createScope('catch', start.line, start.column, end.line, end.column, parentId);
    allScopes[parentId].children.push(scopeId);

    if (node.param?.type === 'Identifier') {
      allScopes[scopeId].bindings.push({
        name: node.param.name,
        line: getPosition(node.param).line,
        column: getPosition(node.param).column,
        type: 'variable',
        isObfuscated: isObfuscatedName(node.param.name),
      });
    }

    scopeStack.push(scopeId);
  },

  // Exit handlers for scope cleanup
  exit(path) {
    const node = path.node;
    const currentScopeId = scopeStack[scopeStack.length - 1];
    const currentScope = allScopes[currentScopeId];

    if (
      node.type === 'FunctionDeclaration' ||
      node.type === 'FunctionExpression' ||
      node.type === 'ArrowFunctionExpression' ||
      node.type === 'ClassDeclaration' ||
      node.type === 'ClassExpression'
    ) {
      if (scopeStack.length > 1) {
        scopeStack.pop();
      }
    }

    // Block statements in loops
    if (
      (path.parent?.type === 'ForStatement' ||
        path.parent?.type === 'ForInStatement' ||
        path.parent?.type === 'ForOfStatement') &&
      node.type === 'BlockStatement'
    ) {
      if (scopeStack.length > 1 && allScopes[scopeStack[scopeStack.length - 1]]?.type === 'block') {
        scopeStack.pop();
      }
    }

    if (node.type === 'CatchClause' && currentScope?.type === 'catch') {
      if (scopeStack.length > 1) {
        scopeStack.pop();
      }
    }
  },
});

// Collect binding reference lines using Babel scope bindings.
// Key format: `${name}:${line}:${column}` where line is 0-indexed.
const bindingReferenceMap = new Map();
function bindingKey(name, line, column) {
  return `${name}:${line}:${column}`;
}

traverse.default(ast, {
  Scope(path) {
    for (const [name, binding] of Object.entries(path.scope.bindings)) {
      const id = binding?.identifier;
      if (!id?.loc) continue;

      const line = id.loc.start.line - 1;
      const column = id.loc.start.column;
      const key = bindingKey(name, line, column);

      if (!bindingReferenceMap.has(key)) {
        bindingReferenceMap.set(key, new Set());
      }

      for (const refPath of binding.referencePaths || []) {
        const refLoc = refPath.node?.loc?.start;
        if (!refLoc) continue;
        bindingReferenceMap.get(key).add(refLoc.line - 1);
      }
    }
  },
});

// Collect all bindings for result
for (const scopeId in allScopes) {
  const scope = allScopes[scopeId];
  const bindingsWithRefs = scope.bindings.map(b => {
    const key = bindingKey(b.name, b.line, b.column);
    const referenceLines = bindingReferenceMap.has(key)
      ? Array.from(bindingReferenceMap.get(key)).sort((a, c) => a - c)
      : [];
    return {
      ...b,
      referenceLines,
    };
  });

  result.scopes[scopeId] = {
    scopeId: scope.scopeId,
    type: scope.type,
    range: scope.range,
    parentId: scope.parentId,
    children: scope.children,
    bindings: bindingsWithRefs,
  };

  // Add to flat bindings list
  bindingsWithRefs.forEach(b => {
    result.bindings.push({
      ...b,
      scopeId: scopeId,
    });
  });
}

// Output JSON
console.log(JSON.stringify(result, null, 2));
