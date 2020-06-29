import ico

i0 = ico.Ico(0)
print(i0.faces[0])
i1 = ico.Ico(1)
print(i1.faces[0])
for c in i1.faces[0].children:
    print(c)
