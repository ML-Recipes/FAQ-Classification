from sentence_transformers import SentenceTransformer, util
import pandas as pd

class Cluster(object):
    """ Class to cluster similar documents 
    
    :param pre_trained_name: BERT pre-trained name
    :param min_threshold: minimum threshold to consider sentence pairs with a cosine-similarity
        larger than threshold as similar
    :param min_community_size: minimum number of documents to appear in a cluster
    :param show_progress_bar: boolean flag to show the progress during embeddings
    :param convert_to_numpy: boolean flag to convert embeddings to numpy format
    """
    def __init__(self, pre_trained_name='distilbert-base-nli-stsb-quora-ranking', min_threshold=0.75, 
                 min_community_size=10, show_progress_bar=True, convert_to_numpy=True, largest=True):
        
        self.largest = largest
        self.min_threshold = min_threshold
        self.convert_to_numpy = convert_to_numpy
        self.show_progress_bar = show_progress_bar
        self.min_community_size = min_community_size
        self.pre_trained_name = pre_trained_name

    def cluster(self, docs):
        """ Generate clusters from list of documents 
        
        :param docs: list of documents
        :return: Python dictionary
        """
        corpus = self.generate_corpus(docs)
        embeddings = self.generate_embeddings(corpus)
        communities = self.community_detection(embeddings)
        clusters = self.get_cluster_docs(communities, corpus)
        return clusters

    def generate_embeddings(self, docs):
        """ Generate sentence embeddings from list of documents 
        
        :param docs: list of documents
        :return: sentence-embeddings
        """
        model = SentenceTransformer(self.pre_trained_name)
        embeddings = model.encode(docs, show_progress_bar=self.show_progress_bar, convert_to_numpy=self.convert_to_numpy)
        return embeddings
    
    def generate_corpus(self, docs):
        """ Generate corpus from list of unique documents
        
        :param docs: list of documents
        :return: unique documents
        """
        docs = list(set(docs))
        return docs
    
    def get_cluster_docs(self, clusters, corpus):
        """ Group similar documents into clusters from
        input corpus documents
        
        :param clusters: communities
        :param corpus: corpus documents
        :return: Python dictionary where
            key  : cluster no
            value: list of documents 
        """
        cluster_docs = dict()

        for cluster_no, cluster in enumerate(clusters):
            
            cluster_no += 1

            docs = []
            for doc_id in cluster:
                docs.append(corpus[doc_id])

            cluster_docs[cluster_no] = docs
        
        return cluster_docs
    
    def get_distribution(self, clusters):
        """ Get the distribution of cluster documents
        
        :param cluster: Python dictionary where
            key  : cluster no
            value: list of documents 
        :return: pandas DataFrame
        """
        df = pd.DataFrame(list(clusters.items()), columns=['Cluster', 'Docs'])
        df['Num_Docs'] = df['Docs'].apply(lambda x: len(x))
        df.drop('Docs', axis=1, inplace=True)
        return df

    def community_detection(self, embeddings):
        """ Extract groups of documents that are highly similar from embeddings
        using Fast Community Detection 
        
        :param embeddings: sentence embeddings
        :return: communities that are larger than min_community_size
        """ 

        # Compute cosine similarity scores
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=self.min_community_size, largest=self.largest)

        # Filter for rows >= min_threshold
        extracted_communities = []
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= self.min_threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=len(embeddings), largest=self.largest)
                top_idx_large = top_idx_large.tolist()
                top_val_large = top_val_large.tolist()

                if top_val_large[-1] < self.min_threshold:
                    for idx, val in zip(top_idx_large, top_val_large):
                        if val < self.min_threshold:
                            break

                        new_cluster.append(idx)
                else:
                    # Iterate over all entries (slow)
                    for idx, val in enumerate(cos_scores[i].tolist()):
                        if val >= self.min_threshold:
                            new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        # Largest cluster first
        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

        # Step 2) Remove overlapping communities
        unique_communities = []
        extracted_ids = set()

        for community in extracted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_communities.append(community)
                for idx in community:
                    extracted_ids.add(idx)

        return unique_communities

    