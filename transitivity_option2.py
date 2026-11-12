import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def analyze_ties_and_cycles(csv_path, task_name):
    print(f"\n{'='*70}")
    print(f"OPTION 2 ANALYSIS — TASK: {task_name.upper()}")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(csv_path)
    # Filter to only successful parses
    df = df[df["parse_ok"].astype(str).str.lower() == "true"]
    
    engines = df["engine"].unique()
    
    for engine in engines:
        eng_df = df[df["engine"] == engine]
        
        # Count ties vs non-ties at the response level
        total_responses = len(eng_df)
        ties = (eng_df["raw_answer"].astype(str).str.upper() == "TIE").sum()
        tie_rate = ties / total_responses
        
        print(f"--- Engine: {engine.upper()} ---")
        print(f"  Total pair comparisons : {total_responses}")
        print(f"  Ties declared          : {ties} ({tie_rate:.1%})")
        print(f"  Non-tied edges created : {total_responses - ties}")
        
        # Now build graphs per composition
        graphs = defaultdict(list)
        for _, row in eng_df.iterrows():
            if str(row["raw_answer"]).upper() == "TIE":
                continue
            comp = row["composition_id"]
            graphs[comp].append((row["doc_a_id"], row["doc_b_id"], row["answer"]))
            
        total_comps = len(graphs)
        comps_with_enough_edges = 0
        comps_with_cycles = 0
        comps_with_enough_and_cycle = 0
        
        for comp_id, edges in graphs.items():
            # 1. Build the directed graph edges (winner -> loser)
            directed_edges = []
            for doc_a, doc_b, answer in edges:
                if answer == doc_a:
                    directed_edges.append((doc_a, doc_b))
                elif answer == doc_b:
                    directed_edges.append((doc_b, doc_a))
                    
            # 2. Check for cycles (Kahn's algorithm)
            adj = defaultdict(set)
            indegree = defaultdict(int)
            nodes = set()
            for w, l in directed_edges:
                adj[w].add(l)
                indegree[l] += 1
                nodes.add(w); nodes.add(l)
                
            queue = [n for n in nodes if indegree[n] == 0]
            visited = 0
            while queue:
                n = queue.pop()
                visited += 1
                for neighbor in adj[n]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
                        
            has_cycle = visited != len(nodes)
            
            # 3. Count stats
            total_comps += 1
            if len(directed_edges) >= 3:
                comps_with_enough_edges += 1
                if has_cycle:
                    comps_with_enough_and_cycle += 1
            if has_cycle:
                comps_with_cycles += 1
                
        # Note: total_comps is just the number of compositions present in the data
        # The number of directed edges is what matters for cycle potential
        
        print(f"  Total compositions      : {total_comps}")
        print(f"  Comps with >= 3 edges   : {comps_with_enough_edges}")
        print(f"  Comps with cycles       : {comps_with_cycles}")
        
        raw_cycle_rate = comps_with_cycles / total_comps if total_comps else 0
        cond_cycle_rate = comps_with_enough_and_cycle / comps_with_enough_edges if comps_with_enough_edges else 0
        
        print(f"  Cycle rate (all comps)  : {raw_cycle_rate:.1%}")
        print(f"  Cycle rate (cond. on >=3 edges): {cond_cycle_rate:.1%}")
        print()

# Run for both tasks
analyze_ties_and_cycles("test2_transitivity_runs_long.csv", "long")
analyze_ties_and_cycles("test2_transitivity_runs_short.csv", "short")