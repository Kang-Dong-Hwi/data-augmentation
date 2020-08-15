# data-augmentation
timestretch

<br><br>
### 1. [data augmentation.ipynb](https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/data%20augmentation.ipynb)
<br>

~~~python
for idx in range( y_data.shape[0] ):
    
    col = binary_search( S_left[:,:,idx] ) +1
    columns.append(col)
~~~


binary_search : zero padding 시작되는 column의 (index -1) 반환 <br>
columns  : (binary_search 반환값 +1)이 저장된 list



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
fixed_rate = ( col / 382 )*0.95 

<br><br><br>

### 2. [inputdata.ipynb](https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/inputdata.ipynb)
<br>

timestretch 후
log scale변환, normalization한 뒤
.pt 로 저장

<br><br><br>
### 3. [DenseNet-aug1.ipynb](https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/DenseNet-aug1.ipynb)


<!--
1. _densenet(growth_rate = 20, block_config = (5,5,5), num_init_features=32)>  : mag, phase
2. _densenet(growth_rate = 64, block_config = (5,5,4), num_init_features=64)>  : mag, phase
3. _densenet(growth_rate = 64, block_config = (5,5,4), num_init_features=64)>  : only mag
-->

epoch=100<br>
batch_size=20<br>
lr=0.00002<br>

<table>

  <tr> 
      <td colspan="4"><br><br>  _densenet(growth_rate = 20, block_config = (5,5,5), num_init_features=32)>  : mag, phase  </td>
  </tr>

  <tr>
    <td> <img src="https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/screenshots/timestretch_train_confusion_matrix.png", height=430px, width=400px> </td>
    <td> <img src="https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/screenshots/time_stretch_train_dataset_confusion_matrix.png", height=430px, width=400px></td>
    
 </tr>
  
  <tr> 
      <td colspan="4">
       training accuracy: 94.625%<br>
       validation accuracy: 67.50%<br>
      </td>
  </tr>
  
  
    
  <tr> 
      <td colspan="4"><br><br> _densenet(growth_rate = 64, block_config = (5,5,4), num_init_features=64)>  : mag, phase </td>
  </tr>

  <tr>
    <td> <img src="https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/screenshots/timestretch_train_confusion_matrix2.png", height=430px, width=400px> </td>
    <td> <img src="https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/screenshots/time_stretch_train_dataset_confusion_matrix2.png", height=430px, width=400px></td>
  </tr>
  
  <tr> 
      <td colspan="4">
       training accuracy: 99.50%<br>
       validation accuracy: 57.5%<br>
      </td>
  </tr>
  
  
    
  <tr> 
      <td colspan="4"><br><br> 3. _densenet(growth_rate = 64, block_config = (5,5,4), num_init_features=64)>  : mag,_  </td>
  </tr>

  <tr>
    <td> <img src="https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/screenshots/timestretch_train_confusion_matrix3.png", height=430px, width=400px> </td>
    <td> <img src="https://github.com/Kang-Dong-Hwi/data-augmentation/blob/master/screenshots/time_stretch_train_dataset_confusion_matrix3.png", height=430px, width=400px></td>
  </tr>
  
  <tr> 
      <td colspan="4">
       training accuracy: 96.75%<br>
       validation accuracy: 88.00%<br>
      </td>
  </tr>
  
  
  
</table>
