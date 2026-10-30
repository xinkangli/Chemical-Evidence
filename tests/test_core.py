import unittest
import numpy as np
from chemical_evidence.bands import self_tests,check_spectrum,summarize_candidate
from chemical_evidence.evaluation import matched_weights
class CoreTests(unittest.TestCase):
 def test_archived_feature_checks(self):self.assertEqual(self_tests()['status'],'passed')
 def test_ties_do_not_use_correctness(self):
  w,t,p=matched_weights([3,2,2,1,2],[True,True,True,True,False],.5)
  np.testing.assert_allclose(w,[1,.5,.5,0,.5]);self.assertEqual(t,2)
 def test_fixed_denominator_at_ps_edge(self):
  x=np.arange(675.,3301.);y=sum(np.exp(-.5*((x-c)/4)**2) for c in [3024,2847,1492,1451,694])
  packet=check_spectrum(x,y,'ATR','absorbance','ps')
  s=summarize_candidate([packet],'polystyrene')
  self.assertEqual(s['expected_checks'],5);self.assertEqual(s['unassessable_checks'],1)
  self.assertLessEqual(s['unique_peak_observations']/s['expected_checks'],.8)
 def test_invalid_scores(self):
  with self.assertRaises(ValueError):matched_weights([float('nan')],[True])
if __name__=='__main__':unittest.main()
