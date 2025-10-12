import numpy as np
from utils import make_ideal_cp, edit_cc, edit_cp, edit_pp

block = [(5, 5), (5, 5)]
cp = make_ideal_cp(block)

print("ideal cp:")
for row in cp:
    print(row)

edited_cp = edit_cp(cp, 0.25)
print("edited cp:")
for row in edited_cp:
    print(row)

edited_cc = edit_cc(cp, block, 0.25)
print("edited cc:")
for row in edited_cc:
    print(row)

# edited_pp = edit_pp(cp, block, 0.25)
# print("edited pp:")
# for row in edited_pp:
#     print(row)