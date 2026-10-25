import numpy as np
from scipy import sparse
from media_gnn_study.models import prune_tfidf_topk, prepare_hetero_term_incidence

def test_topk_pruning_is_row_bounded():
    x=sparse.csr_matrix(np.array([[1.,4.,2.,3.],[0.,1.,0.,2.]],dtype=float))
    y=prune_tfidf_topk(x,2)
    assert max(np.diff(y.indptr)) <= 2
    assert set(y[0].indices.tolist()) == {1,3}

def test_hetero_incidence_normalizes_without_labels():
    tr=sparse.csr_matrix(np.array([[1.,1.],[0.,2.]],dtype=float))
    va=sparse.csr_matrix(np.array([[3.,1.]],dtype=float))
    row_tr,col_tr,row_va,meta=prepare_hetero_term_incidence(tr,va)
    assert np.allclose(row_tr.sum(1),1)
    assert np.allclose(row_va.sum(1),1)
    assert meta['term_nodes']==2
