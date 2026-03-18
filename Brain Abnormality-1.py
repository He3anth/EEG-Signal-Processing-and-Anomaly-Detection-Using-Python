import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# read EEG file (single column data)
path = "C:/Hemanth/Python Lab/Brain Abnormality/adhdata.csv"
data = pd.read_csv(path)

sig = data.iloc[:,0].values   # takes only first column
Fs = 256   # assuming standard sampling frequency


#bandpass filter
def filt(x):
    nyq = 0.5*Fs
    low = 0.5/nyq
    high = 50/nyq

    b,a = butter(5,[low,high],btype='band')
    y = lfilter(b,a,x)
    return y

f_sig = filt(sig)


# feature extraction (basic ones only)
def feat_ext(s):
    f = []

    f.append(np.mean(s))
    f.append(np.std(s))
    f.append(np.var(s))

    # checking peaks also
    f.append(np.max(s))
    f.append(np.min(s))

    fr, ps = welch(s,Fs)

    # frequency bands (approx)
    d = np.sum(ps[(fr>=0.5)&(fr<4)])
    t = np.sum(ps[(fr>=4)&(fr<8)])
    a = np.sum(ps[(fr>=8)&(fr<13)])
    b = np.sum(ps[(fr>=13)&(fr<30)])

    f.extend([d,t,a,b])

    # randomness measure
    f.append(entropy(ps))

    return f


# divide signal into windows
w = Fs*2
X = []
y = []

i = 0
while i < len(f_sig)-w:
    seg = f_sig[i:i+w]

    X.append(feat_ext(seg))

    # rough condition (not exact)
    if np.var(seg) > 1000:
        y.append(1)
    else:
        y.append(0)

    i = i + w


X = np.array(X)
y = np.array(y)


# split data (no random_state used intentionally)
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2)

# model
rf = RandomForestClassifier(100)
rf.fit(Xtr,ytr)

pred = rf.predict(Xte)

print("\nReport:\n")
print(classification_report(yte,pred))


# final decision
ab = sum(pred)
tot = len(pred)

per = (ab/tot)*100

print("\nSegments:",ab,"/",tot)
print("Abnormal %:",round(per,2))

if per>30:
    print("EEG is Abnormal")
else:
    print("EEG is Normal")


# plot small portion
plt.plot(f_sig[:1000])
plt.title("EEG filtered")
plt.xlabel("samples")
plt.ylabel("amp")
plt.show()


# showing each segment result
print("\nSegments info:")
for i in range(len(pred)):
    if pred[i]==1:
        print("seg",i,"abnormal")
    else:
        print("seg",i,"normal")
