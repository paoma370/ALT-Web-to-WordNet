"""Reproduce v0.2 with Python 3 standard library: python3 scripts/analyze.py."""
import csv, gzip, hashlib, json, platform, re, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict, deque
from itertools import combinations
R=Path(__file__).resolve().parents[1]; O=R/'results';O.mkdir(exist_ok=True)
def output(name,rows):
    if rows:
        with (O/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
raw=list(csv.DictReader((R/'input/mapping_completo.csv').open(encoding='utf-8-sig',newline='')))
assert all(r['semanticnet_settore']=='agricoltura' for r in raw)
root=ET.fromstring(gzip.open(R/'input/wnita.lmf.gz','rb').read())
sx=root.findall('.//Synset'); ids=[s.get('id') for s in sx]; syn=set(ids)
assert len(ids)==len(syn), 'Duplicate Synset IDs'
gloss={s.get('id'):'; '.join(d.get('gloss','') for d in s.findall('Definition')) for s in sx}
lemmas=defaultdict(set)
for entry in root.findall('.//LexicalEntry'):
    lemma=entry.find('Lemma')
    for sense in entry.findall('Sense'):
        if lemma is not None:lemmas[sense.get('synset')].add(lemma.get('writtenForm',''))
rels=set((s.get('id'),r.get('relType',''),r.get('target')) for s in sx for r in s.findall('.//SynsetRelation'))
relation_index=defaultdict(set)
for s,r,t in rels:relation_index[(s,t)].add(r)
explicit={(s,t) for s,r,t in rels if r=='has_hyperonym'}
inverse={(t,s) for s,r,t in rels if r=='has_hyponym'}
edges=explicit|inverse
dangling={(s,t) for s,t in edges if s not in syn or t not in syn}
parents=defaultdict(set)
for s,t in edges:
    if s in syn and t in syn:parents[s].add(t)
def ancestors(s):
    d={s:0};todo=deque([s])
    while todo:
        x=todo.popleft()
        for p in parents[x]:
            if p not in d:d[p]=d[x]+1;todo.append(p)
    return d
def pos(s):return s.rsplit('-',1)[-1]
def dist(a,b,cache):
    common=set(cache[a])&set(cache[b])
    if not common:return None,[]
    d=min(cache[a][x]+cache[b][x] for x in common)
    return d,sorted(x for x in common if cache[a][x]+cache[b][x]==d)
# Kahn diagnostic: residual includes cycles and nodes downstream from cycles.
indeg=Counter(t for s,ts in parents.items() for t in ts)
queue=deque(s for s in syn if not indeg[s]);visited=set()
while queue:
    s=queue.popleft();visited.add(s)
    for t in parents[s]:
        indeg[t]-=1
        if not indeg[t]:queue.append(t)
residual=syn-visited
rows=[r for r in raw if r['semanticnet_tpdom']=='O'];groups=defaultdict(set);concepts={};cm=defaultdict(set); micro=defaultdict(set)
for r in rows:
    q=r['semanticnet_FK_question_id'];m=r['semanticnet_chiave'];groups[m].add(q)
    direct=r['semanticnet_parola_WN'];kind='direct' if direct and direct!='NA' else 'closest'
    t=direct if kind=='direct' else r['semanticnet_parola_WN_closest'];cm[q].add((t,kind))
    concepts[q]={'question_id':q,'question_number':r['semanticnet_ndom'],'label':r['semanticnet_parola'],'text':r['semanticnet_testo'],'target':t,'mapping':kind,'pos':pos(t)}
for r in raw:
    for k,kind in [('semanticnet_chiave_WN_1','primary'),('semanticnet_chiave_WN_2','alternative2'),('semanticnet_chiave_WN_3','alternative3'),('semanticnet_chiave_WN_closest','closest')]:
        if r[k].startswith('ita-'):micro[r['semanticnet_chiave']].add((r[k],kind))
assert all(len(x)==1 for x in cm.values())
macro=set(r['semanticnet_settore_WN'] for r in raw if r['semanticnet_settore_WN'])
targets={c['target'] for c in concepts.values()}|{t for v in micro.values() for t,k in v}|macro
missing=targets-syn
assert not missing, 'Missing mapped synsets: '+str(missing)
anc={t:ancestors(t) for t in targets}
relevant_residual=residual&{a for d in anc.values() for a in d}
output('synset_inventory.csv',[{'id':s,'pos':pos(s),'lemmas':'; '.join(sorted(lemmas[s])),'gloss':gloss[s]} for s in sorted(syn)])
output('concepts.csv',list(sorted(concepts.values(),key=lambda c:c['question_id'])))
output('alt_memberships.csv',[{'microclass':m,'question_id':q} for m in sorted(groups) for q in sorted(groups[m])])
output('mapped_synsets.csv',[{'id':s,'lemmas':'; '.join(sorted(lemmas[s])),'gloss':gloss[s]} for s in sorted(targets)])
pairrows=[]
for a,b in combinations(sorted(concepts),2):
    ca,cb=concepts[a],concepts[b];samepos=ca['pos']==cb['pos'];common=sorted(m for m,v in groups.items() if a in v and b in v)
    d,lca=dist(ca['target'],cb['target'],anc) if samepos else (None,[])
    pairrows.append({'q1':a,'q2':b,'label1':ca['label'],'label2':cb['label'],'share_alt_group':bool(common),'shared_microclasses':';'.join(common),'same_pos':samepos,'both_direct':ca['mapping']==cb['mapping']=='direct','same_target':ca['target']==cb['target'],'ancestor_path_distance':d if d is not None else '','closest_common_ancestors':';'.join(lca)})
output('concept_pairs.csv',pairrows)
overlaps=[]
for a,b in combinations(sorted(groups),2):
    inter=groups[a]&groups[b];union=groups[a]|groups[b]
    overlaps.append({'microclass1':a,'microclass2':b,'n1':len(groups[a]),'n2':len(groups[b]),'intersection':len(inter),'jaccard':len(inter)/len(union),'identical_membership':groups[a]==groups[b]})
output('microclass_overlaps.csv',overlaps)
members=[]
for m,qs in sorted(groups.items()):
    for q in sorted(qs):
        c=concepts[q]
        for t,kind in sorted(micro[m]):
            relation='same_synset' if t==c['target'] else 'concept_is_descendant' if t in anc[c['target']] else 'microclass_is_descendant' if c['target'] in anc[t] else 'no_ancestor_relation'
            typed=sorted(relation_index[(c['target'],t)]-{'has_hyperonym','has_hyponym'})
            members.append({'microclass':m,'micro_target':t,'micro_mapping':kind,'question_id':q,'concept':c['label'],'concept_target':c['target'],'concept_mapping':c['mapping'],'same_pos':pos(t)==c['pos'],'taxonomic_status':relation,'direct_typed_relations_concept_to_micro':';'.join(typed)})
output('microclass_concept_relations.csv',members)
macrorows=[]
for m,ms in sorted(micro.items()):
    for t,kind in sorted(ms):
        for ma in sorted(macro):
            macrorows.append({'microclass':m,'micro_target':t,'mapping':kind,'macro_target':ma,'status':'same_synset' if t==ma else 'microclass_is_descendant' if ma in anc[t] else 'macroclass_is_descendant' if t in anc[ma] else 'no_ancestor_relation'})
output('macro_micro_relations.csv',macrorows)
tar=defaultdict(list)
for q,c in concepts.items():tar[c['target']].append(q)
typedrows=[]
for s,r,t in sorted(rels):
    if s in tar and t in tar and r not in ('has_hyperonym','has_hyponym'):
        for a in tar[s]:
            for b in tar[t]:
                if a!=b:typedrows.append({'q1':a,'q2':b,'label1':concepts[a]['label'],'label2':concepts[b]['label'],'source':s,'relation':r,'target':t,'both_direct':concepts[a]['mapping']==concepts[b]['mapping']=='direct','share_alt_group':any(a in v and b in v for v in groups.values())})
output('typed_relations_between_concepts.csv',typedrows)
output('target_ancestors.csv',[{'target':s,'ancestor':a,'steps':d} for s in sorted(anc) for a,d in sorted(anc[s].items())])
relevant_nodes={a for d in anc.values() for a in d}
output('relevant_hierarchy_edges.csv',[{'child':a,'parent':b} for a,b in sorted(edges) if a in relevant_nodes and b in relevant_nodes])
output('relevant_roots.csv',[{'target':s,'root':a,'steps':d,'root_lemmas':'; '.join(sorted(lemmas[a])),'root_gloss':gloss[a]} for s in sorted(anc) for a,d in sorted(anc[s].items()) if not parents[a]])
output('inverse_asymmetries.csv',[{'child':a,'parent':b,'hyperonym_asserted':(a,b) in explicit,'hyponym_asserted':(a,b) in inverse} for a,b in sorted(explicit^inverse)])
old=(R/'input/previous_subgraph.ttl').read_text();oldedges=set(re.findall(r'^wn:(\S+) rdfs:subClassOf wn:(\S+) \.',old,re.M));oldparents=defaultdict(set)
for a,b in oldedges:oldparents[a].add(b)
def oldanc(s):
    seen={s:0};todo=deque([s])
    while todo:
        x=todo.popleft()
        for p in oldparents[x]:
            if p not in seen:seen[p]=seen[x]+1;todo.append(p)
    return seen
oldcache={t:oldanc(t) for t in tar}
comparison=[]
for p in pairrows:
    if p['same_pos']:
        a=concepts[p['q1']]['target'];b=concepts[p['q2']]['target'];d,_=dist(a,b,oldcache)
        comparison.append({'q1':p['q1'],'q2':p['q2'],'label1':p['label1'],'label2':p['label2'],'old_distance':d if d is not None else '','full_distance':p['ancestor_path_distance'],'changed':(d if d is not None else '')!=p['ancestor_path_distance']})
output('comparison_v01_v02_pairs.csv',comparison)
primary=[r for r in members if r['micro_mapping'] in ('primary','closest')]
primary_direct=[r for r in primary if r['micro_mapping']=='primary' and r['concept_mapping']=='direct']
summary={'version':'0.2','lexicon_attributes':root.find('Lexicon').attrib,'synsets':len(syn),'lexical_entries':len(root.findall('.//LexicalEntry')),'unique_relations':len(rels),'relation_counts':dict(Counter(r for s,r,t in rels)),'hypernym_explicit':len(explicit),'hypernym_from_inverse':len(inverse),'hypernym_union':len(edges),'inverse_only_edges':len(inverse-explicit),'explicit_only_edges':len(explicit-inverse),'dangling_taxonomy_edges':len(dangling),'all_dangling_relations':sum(s not in syn or t not in syn for s,r,t in rels),'cycle_residual_nodes':len(residual),'mapped_ancestor_cycle_residual':len(relevant_residual),'cross_pos_taxonomy_edges':sum(pos(s)!=pos(t) for s,t in edges),'missing_mapped_ids':sorted(missing),'mapped_ids':len(targets),'source_rows':len(raw),'O_rows':len(rows),'concepts':len(concepts),'microclasses_O':len(groups),'microclasses_all':len({r['semanticnet_chiave'] for r in raw}),'mapping_counts':dict(Counter(c['mapping'] for c in concepts.values())),'distinct_concept_targets':len(tar),'identical_microclass_pairs':[r for r in overlaps if r['identical_membership']],'memberships_taxonomic_status_primary_or_closest':dict(Counter(r['taxonomic_status'] for r in primary)),'both_direct_memberships':len(primary_direct),'both_direct_membership_status':dict(Counter(r['taxonomic_status'] for r in primary_direct)),'macro_micro_primary_status':dict(Counter(r['status'] for r in macrorows if r['mapping'] in ('primary','closest'))),'changed_pair_distances':sum(r['changed'] for r in comparison),'old_taxonomy_edges_absent_from_full':len(oldedges-edges),'typed_relation_rows':len(typedrows),'pair_descriptives':[]}
for mode in ('direct_only','all_mappings'):
    for share in (True,False):
        ps=[p for p in pairrows if p['same_pos'] and p['share_alt_group']==share and (mode=='all_mappings' or p['both_direct'])];ds=[p['ancestor_path_distance'] for p in ps if p['ancestor_path_distance']!='']
        summary['pair_descriptives'].append({'mode':mode,'share_alt_group':share,'same_pos_pairs':len(ps),'connected_pairs':len(ds),'mean_distance_connected':sum(ds)/len(ds) if ds else None})
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
manifest={'version':'0.2','python':platform.python_version(),'inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((R/'input').iterdir())},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='relation_counts'},ensure_ascii=False,indent=2))
