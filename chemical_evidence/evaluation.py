"""Retrospective matching with label-independent fractional treatment of ties."""
import numpy as np

def matched_weights(scores, known, target=.9):
    scores=np.asarray(scores,dtype=float); known=np.asarray(known,dtype=bool)
    if scores.ndim!=1 or known.shape!=scores.shape or not known.any():
        raise ValueError('Need aligned one-dimensional scores and a nonempty known set')
    if not np.isfinite(scores).all() or not 0<target<=1:
        raise ValueError('Finite scores and 0 < target <= 1 required')
    wanted=target*known.sum()
    threshold=np.sort(scores[known])[::-1][int(np.ceil(wanted))-1]
    above=scores>threshold; ties=scores==threshold
    probability=float((wanted-(above&known).sum())/(ties&known).sum())
    weights=above.astype(float)+ties*probability
    return weights,float(threshold),probability

def metrics(scores,known,correct,target=.9):
    known=np.asarray(known,dtype=bool); correct=np.asarray(correct,dtype=bool)
    if correct.shape!=known.shape:raise ValueError('Misaligned correctness array')
    w,t,p=matched_weights(scores,known,target)
    return dict(known_acceptance=float(w[known].mean()),
        unknown_acceptance=float(w[~known].mean()) if (~known).any() else None,
        accepted_known_error=float((w[known]*~correct[known]).sum()/w[known].sum()),
        total_rejection=float(1-w.mean()),threshold=t,boundary_probability=p)
