import torch
import pickle

__GENE_FILE = 'test.txt'
init_graph_path = 'new_score.pkl'

feature_dim = 20499

if __name__ == "__main__":
    init_relation_graph = torch.eye(feature_dim)
    init_relation_graph = init_relation_graph * 999
    init_relation_graph.requires_grad = False

    gene_list = []

    with open(__GENE_FILE, 'r') as gf:
        for line in gf.readlines():
            gene = int(line[:-1])
            gene_list.append(gene)
    print('opening gene list done')

    with open(init_graph_path, 'rb') as f:
        score_dict = pickle.loads(f.read())
    print('opening score dict done')

    count = 0

    for gene_pair in score_dict:
        gene1 = gene_pair[0]
        gene2 = gene_pair[1]
        print('gene1', gene1)
        gene1_idx = gene_list.index(gene1)
        gene2_idx = gene_list.index(gene2)
        score = score_dict[gene_pair]
        # rule out irrevelent two genes
        init_relation_graph[gene1_idx, gene2_idx] = score
        print(gene1, gene2, score)
        count += 1
        print(count)

    print(init_relation_graph)
    print(torch.max(init_relation_graph))

    init_relation_graph = torch.softmax(init_relation_graph, dim=1)

    print(init_relation_graph)

    torch.save(init_relation_graph, 'init graph.pth')
