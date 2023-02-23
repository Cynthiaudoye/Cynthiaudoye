s1='International'
s2='Airport'
print("Original strings:\n%s \n%s " %(s1, s2))

s3=s1+' '+s2
print(' Concatenated strings: ', s3)

s4=s3.replace(" ", "")
print(' Same but with space removed: ', s4)

s5=s3.replace("o", "0")
print(' Same but with letters o replaced by 0: ', s5)

print(' Selecting specific symbols in a string: ', s3[14:17])
A = 'Interpretation'
B = 'of result'
C = A + ' ' + B
print(C)
print(A.replace("Interpretation", "Interpreting"))