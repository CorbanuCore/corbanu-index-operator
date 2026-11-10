// Explicitly freeze every portfolio choice; no helper default is relied upon.
import {readFileSync, writeFileSync} from 'node:fs';
import {capWeights} from './weights.ts';
const [manifestPath, scoresPath, output] = process.argv.slice(2);
if (!manifestPath || !scoresPath || !output) throw Error('Usage: weights.mts MANIFEST SCORES OUTPUT');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const scores = JSON.parse(readFileSync(scoresPath, 'utf8'));
if (!Array.isArray(scores) || scores.some((r:any)=>!r || typeof r.security_id !== 'string' || !Number.isInteger(r.score) || r.score < 1 || r.score > 100)) {
  throw Error('Every frozen security requires an integer score in 1..100');
}
const result = capWeights(manifest, scores, 70, 'market_cap_rank', 'exclude');
writeFileSync(output, JSON.stringify(result, null, 2) + '\n', {flag:'wx'});
console.log(JSON.stringify({status:result.status,included:result.included_count,excluded:result.excluded_count}));
