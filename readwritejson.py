import json

f = open('caption.json', 'r')
data = json.load(f)
print(data)
f.close()

s = data['lastnumcaption'] + 1
f = open('caption.json', 'w')
data['lastnumcaption'] = s
json.dump(data, f)
f.close()

s1 = data['lastnumtemplate'] + 1
f = open('caption.json', 'w')
data['lastnumtemplate'] = s1
json.dump(data, f)
f.close()