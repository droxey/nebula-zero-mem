import networkx as nx
from zero_mem.index_builder import Entity, Message, ZeroMemIndex
from zero_mem.retriever import parse_query, RetrievedEvidence, evidence_closure
from zero_mem.calibration import hard_filter

class FakeNLP:
    def __call__(self, text):
        class T:
            def __init__(self, x):
                self.text=x; self.lemma_=x.lower(); self.pos_="NOUN"; self.is_stop=x.lower() in {"what","the","and"}; self.is_punct=False; self.like_num=False
        class D:
            ents=[]
            def __iter__(self): return iter([T(x) for x in text.replace('?','').split()])
        return D()

def idx():
    msgs=[Message("1","t","user","docker caprover backup migration",1,0,{"thread_name":"t"})]
    g=nx.Graph(); g.add_nodes_from(["PERSON:docker", "ORG:caprover", "ORG:backup", "ORG:migration"])
    return ZeroMemIndex(msgs,[],g,{"local":[],"episode":[],"window":[],"turn":[]},None,"",1,{})

def test_intent_distinguishes_relational_local_mixed():
    i=idx(); n=FakeNLP()
    assert parse_query("Docker and CapRover together",i,n).intent == "relational"
    assert parse_query("What happened recently about backup",i,n).intent == "local"
    assert parse_query("Compare Docker with CapRover recently",i,n).intent == "mixed"

def test_keywords_exclude_stopwords_and_entities():
    p=parse_query("What is the backup migration project?",idx(),FakeNLP())
    assert "backup" in p.keywords and "migration" in p.keywords
    assert "what" not in p.keywords and "the" not in p.keywords

def test_evidence_closure_uses_both_views():
    i=idx(); out=evidence_closure([(0,.8)],[(0,.7)],i,rho=.5)
    assert out and out[0].source_view == "both"

def test_hard_filter_excludes_subject_mismatch():
    m=idx().messages[0]
    ev=RetrievedEvidence(m,.5,"graph",related_entities=[Entity("Other","ORG","1","t",0,5,0)])
    assert hard_filter([ev],["ORG:docker"]) == []
