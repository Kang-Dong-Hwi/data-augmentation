# data-augmentation
timestretch


#### [data augmentation.ipynb](https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/data%20augmentation.ipynb)
<br><br>

~~~python
for idx in range( y_data.shape[0] ):
    
    col = binary_search( S_left[:,:,idx] ) +1
    columns.append(col)
~~~

<!--
binary_search : zero padding 시작되는 column의 (index -1) 반환
columns  : (binary_search 반환값 +1)이 저장된 list
-->


<br>
<br>

~~~python

hop_length = 250
NFFT = 512
n_freq = (NFFT//2)-1


for idx in range( y_data.shape[0] ):

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


$$\fixed_rate = { col \over 382 }*0.95$$

\begin{equation}
y = x^2
\end{equation}

