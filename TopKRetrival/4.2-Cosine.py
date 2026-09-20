import math

a=[1,2,3]
b=[4,5,6]

ans=0
for j in range(len(a)):
    ans += a[j]*b[j]
print("Dot:" ,ans)

sqa=0
for i in a:
    sqa += i*i
sqa=math.sqrt(sqa)
print("mag a:", sqa)

sqb=0
for i in b:
    sqb +=i*i
sqb=math.sqrt(sqb)
print("mag b:", sqb)

print(ans/(sqa*sqb))