// Read-only comparison with the pinned original implementation. No provider calls.
import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {capWeights} from './weights.ts';
const read = (p:string) => JSON.parse(readFileSync(p, 'utf8'));
const [lockPath, vectorsPath, manifestPath, scoresPath, expectedPath] = process.argv.slice(2);
for (const file of read(lockPath).files) {
  assert.equal(createHash('sha256').update(readFileSync(file.path)).digest('hex'), file.sha256, 'Source drift: '+file.path);
}
const {legacyMessages, finalizerMessages, validateBrief} = await import('/home/pfrpc/repos/CorbanuAPI/src/indexing/legacy.ts');
const {canonical} = await import('/home/pfrpc/repos/CorbanuAPI/src/indexing/common.ts');
const {capWeights: originalWeights} = await import('/home/pfrpc/repos/CorbanuAPI/src/indexing/weights.ts');
const vectors = read(vectorsPath);
for (const v of vectors.messages) {
  assert.deepEqual(v.expected, v.stage === 'capped_finalizer' ? finalizerMessages(v.context,v.notes) : legacyMessages(v.context));
}
for (const v of vectors.canonical) assert.equal(canonical(v.input).toString('utf8'),v.expected);
for (const v of vectors.briefs) assert.deepEqual(validateBrief(JSON.parse(v.raw)),v.expected);
const manifest = read(manifestPath), scores = read(scoresPath);
const result = capWeights(manifest,scores,70,'market_cap_rank','exclude');
assert.deepEqual(result, originalWeights(manifest,scores,70,'market_cap_rank','exclude'));
assert.deepEqual(result, read(expectedPath).construction);
assert.equal(result.weights.reduce((n:number,r:any)=>n+r.weight_units,0),1_000_000_000_000);
assert.equal(result.included_count,29);
assert.deepEqual(result.excluded.map((r:any)=>r.ticker).sort(),['LIT','REMX']);
assert.throws(()=>capWeights(manifest,scores.slice(1),70,'market_cap_rank','exclude'));
assert.throws(()=>capWeights(manifest,[...scores.slice(1),scores[1]],70,'market_cap_rank','exclude'));
assert.equal(capWeights(manifest,scores.map((r:any)=>({...r,score:1})),70,'market_cap_rank','exclude').status,'needs_data');
const allMissing = structuredClone(manifest);
for (const row of allMissing.securities) row.capitalizations = {};
assert.equal(capWeights(allMissing,scores,70,'market_cap_rank','exclude').status,'needs_data');
console.log(JSON.stringify({source_locks_verified:true,cross_language_vectors:true,construction_equal:true,holdings:29,excluded:['LIT','REMX'],coverage_and_empty_result_checks:true,publication_allowed:false}));
