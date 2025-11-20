import Mathlib
import Aesop

-- Implementation
def cubeElements (a : Array Int) : Array Int :=
  -- << CODE START >>
  x
  -- << CODE END >>


-- Theorem: The length of the output array must be the same as the length of the input array; Each element in the output array is the cube of the corresponding element in the input array
def cubeElements_spec (a : Array Int) (result : Array Int) : Prop :=
  -- << SPEC START >>
  (result.size = a.size) ∧
  (∀ i, i < a.size → result[i]! = a[i]! * a[i]! * a[i]!)
  -- << SPEC END >>

theorem cubeElements_spec_satisfied (a : Array Int) :
  cubeElements_spec a (cubeElements a) := by
  -- << PROOF START >>
  unfold cubeElements cubeElements_spec
  rfl
  -- << PROOF END >>


#guard cubeElements (#[1, 2, 3, 4]) = (#[1, 8, 27, 64])
#guard cubeElements (#[0, -1, -2, 3]) = (#[0, -1, -8, 27])
#guard cubeElements (#[]) = (#[])
#guard cubeElements (#[5]) = (#[125])
#guard cubeElements (#[-3, -3]) = (#[-27, -27])