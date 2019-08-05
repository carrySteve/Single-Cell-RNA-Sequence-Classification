import sys

import pickle

__SCORE_FILE = 'dataset/10090.protein.links.v11.0.txt'
__GENE_FILE = 'test.txt'

if __name__ == "__main__":
    gene_list = []
    with open(__GENE_FILE, 'r') as gf:
        for line in gf.readlines():
            gene = int(line[:-1])
            gene_list.append(gene)
    
    score_dict = {}

    match = 0
    with open(__SCORE_FILE, 'r') as sf:\
        for lidx, line in enumerate(sf.readlines()):
            values = line[:-1].split(' ')
            if lidx < 1:
                continue
            gene1 = int(values[0].split('ENSMUSP')[-1])
            gene2 = int(values[1].split('ENSMUSP')[-1])
            score = int(values[2])

            if lidx % 100000 == 0:
                print(lidx)

            if gene1 in gene_list: 
                if gene2 in gene_list:
                    print(gene1, gene2, score)
                    score_dict[(gene1, gene2)] = score
                    match += 1
                    # break

    print(match)

    with open('new_score.pkl', 'wb+') as sf:
        pickle.dump(score_dict, sf)
