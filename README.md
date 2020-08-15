# data-augmentation
timestretch



#### [data augmentation.ipynb](https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/data%20augmentation.ipynb)

<br><br>

~~~python

def binary_search( stft ):
columns = []
~~~


<br>
~~~python

col = binary_search( S_left[:,:,idx] ) +1
columns.append(col)

~~~


<br>
~~~python

hop_length = 250

NFFT = 512
n_freq = (NFFT//2)-1

fixed_rate = math.ceil( columns[idx] / 382 * 0.95 * 100 ) / 100

aug1 = nn.Sequential(
        transforms.TimeStretch( 
                hop_length=hop_length, 
                n_freq=n_freq, 
                fixed_rate=fixed_rate 
         ))

aug1 = aug1.cuda()
out_left = aug( STFT_left[idx] )

~~~

<!--
columns :
fixed_rate : 


-->
