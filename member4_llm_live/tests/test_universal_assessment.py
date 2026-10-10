from src.universal_assessment import split_coverage, merge_estimates

def test_coverage_split():
    rows = [{'esco_id':'a','coverage':'assessable'}, {'esco_id':'b','coverage':'no_question_bank'}, {'esco_id':'c','coverage':'unknown_esco_id'}]
    quizzes, ratings, unknown = split_coverage(rows)
    assert quizzes == ['a'] and [r['esco_id'] for r in ratings] == ['b'] and [r['esco_id'] for r in unknown] == ['c']

def test_merge_labels_sources():
    rows = [{'esco_id':'a','coverage':'assessable'}, {'esco_id':'b','coverage':'no_question_bank'}, {'esco_id':'c','coverage':'unknown_esco_id'}]
    estimates, metadata = merge_estimates({'a':0.8}, {'b':0.5, 'c':1.0}, rows)
    assert estimates == {'a':.8,'b':.5}
    assert metadata['b']['method']=='self_report' and metadata['c']['method']=='unassessed'

def test_no_missing_assumed_zero():
    estimates, metadata = merge_estimates({}, {}, [{'esco_id':'b','coverage':'no_question_bank'}])
    assert estimates == {} and metadata['b']['estimate'] is None
