import numpy as np
from media_gnn_study.models import build_knn_propagation, build_knn_edge_index

def test_fixed_knn_rows_sum_to_one_and_no_validation_to_train():
    tr=np.array([[1.,0.],[0.9,0.1],[0.,1.]],dtype=np.float32)
    va=np.array([[0.8,0.2],[0.1,0.9]],dtype=np.float32)
    p,pe,_=build_knn_propagation(tr,va,k=2,self_weight=.9)
    assert np.allclose(p.sum(1),1)
    assert np.allclose(pe.sum(1),1)
    n=len(tr)
    assert not pe[:n,n:].any()

def test_pyg_edge_index_validation_destinations_only_receive_from_train():
    tr=np.array([[1.,0.],[0.9,0.1],[0.,1.]],dtype=np.float32)
    va=np.array([[0.8,0.2],[0.1,0.9]],dtype=np.float32)
    edge=build_knn_edge_index(tr,va,k=2).cpu().numpy()
    n=len(tr)
    src,dst=edge
    # quaisquer arestas destinadas a validação têm origem exclusivamente no treino
    assert np.all(src[dst>=n] < n)
    # não há mensagem saindo de validação
    assert np.all(src < n)
