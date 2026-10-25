
from .config import *
from .data import build_dataset_context, fold_summary
from .features import prepare_feature_context
from .plan import build_experiment_plan, plan_summary
from .sanity import run_sanity_checks
from .engine import execute_experiments
from .diagnostics import run_diagnostics
from .study import assemble_study_result
from .render import persist_study_result, render_study
from .manifest import write_manifest

def run_study(render=True):
    ensure_runtime_dirs()
    dataset=build_dataset_context()
    plan=build_experiment_plan(dataset)
    print('Dataset:',len(dataset.frame),'development | audit reservado:',dataset.metadata['n_audit_reserved'])
    print('Plano:',plan_summary(plan,dataset))
    print('1/5 — preparando representações')
    feature_context=prepare_feature_context(dataset)
    print('2/5 — sanity checks')
    assert run_sanity_checks(dataset,feature_context,plan)
    print('3/5 — executando matriz + experimentos focais')
    experiment_result=execute_experiments(dataset,feature_context,plan)
    print('4/5 — diagnósticos e StudyResult')
    diagnostics=run_diagnostics(dataset,feature_context,experiment_result,plan)
    study_result=assemble_study_result(dataset,experiment_result,diagnostics)
    report_dir=STUDY_DIR/'report'
    persist_study_result(study_result,report_dir)
    figures=[]
    if render:
        print('5/5 — apresentação')
        figures=render_study(study_result,report_dir)
    write_manifest(dataset,feature_context,experiment_result,study_result,diagnostics,plan)
    return {'dataset':dataset,'plan':plan,'features':feature_context,'experiments':experiment_result,
            'diagnostics':diagnostics,'study':study_result,'figures':figures,'run_dir':RUN_DIR}
