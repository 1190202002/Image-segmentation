<a name="IbkzH"></a>
## 1 阈值法
<a name="BxBFX"></a>
### 1.1 算法简介
根据输入图像统计生成一维灰度直方图。通过分析图像灰度的变化，确定最优二值化的值，将图像分为前景和背景**。**以下图为例说明。<br />![Original_screenshot_16.12.2023.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1702715835908-8e209ae0-54f9-4b00-a2a9-ef58473ab299.png#averageHue=%233b3b3b&clientId=u1993a3fe-0925-4&from=paste&height=190&id=n8vsG&originHeight=380&originWidth=362&originalType=binary&ratio=2&rotation=0&showTitle=false&size=144966&status=done&style=none&taskId=u0b51f1e1-9061-48a6-a8a2-cc12ddc05cf&title=&width=181)<br />使用统计方法，统计每个灰度值的像素点数，如下图所示，为原图的一维灰度直方图。阈值法找到一个灰度阈值，以阈值为分界将小于阈值的像素灰度设为0，大于的设为255。即将图像分为前景和背景的二值化图。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1702715900815-c80ffca2-de82-46d0-9b49-99875fa83d4b.png#averageHue=%23fcfcfc&clientId=u1993a3fe-0925-4&from=paste&height=240&id=FyRbU&originHeight=480&originWidth=640&originalType=binary&ratio=2&rotation=0&showTitle=false&size=16036&status=done&style=none&taskId=u1a909d3e-6393-4fba-86f3-cad3d48a36e&title=&width=320)<br />同样，也可使用二维灰度直方图，引入局部空间信息，进行分割。二维灰度直方图灰度为x轴，邻域平均像素为y轴，统计落于x，y的像素点个数。阈值法找到适当的x，y。将图像分为前景和背景。下图为原图的二维灰度直方图。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1702716404530-a2cf6930-f050-47ca-8c01-181d6da10576.png#averageHue=%23aa8db1&clientId=u1993a3fe-0925-4&from=paste&height=240&id=b8uTC&originHeight=480&originWidth=640&originalType=binary&ratio=2&rotation=0&showTitle=false&size=23739&status=done&style=none&taskId=u9c61eed2-949f-4674-9321-80b69d70100&title=&width=320)<br />如何通过数学或模型的方法取得最优二值化的值，决定了阈值法分割结果的好坏。本文主要使用最大类间方差法、最大熵法进行阈值估计。<br />**最大类间方差法算法思想：** 选择灰度值将灰度图分为两类，当类间方差最大时，即为最优分割。<br />**最大熵方法算法思想：**选择灰度值将灰度图分为两类，当两类的总熵最大时，即为最优分割。最大熵方法根据图像的信息熵来选择阈值，以使得分割后的两部分图像具有最大的信息熵。
<a name="rTTlT"></a>
### 1.2 算法实现
<a name="JnBJQ"></a>
#### 最大类间方差的阈值选择法（OTSU）
1. 统计图像的灰度直方图。<br />2. 遍历灰度直方图中每个灰度值作为阈值，阈值将灰度小于和大于它的分为两类。计算类间方差，用来衡量在特定阈值下图像的分离程度，公式为类间方差 = w0 * w1 * (μ0 - μ1)2。其中，w0和w1分别是分割结果中两个区域的像素比例，μ0和μ1分别是两个区域的灰度平均值。<br />3. 选择类间方差最大的阈值作为最终阈值。这个阈值被用来分割图像，将图像分为背景和前景两部分。
<a name="BheO8"></a>
#### 最大熵方法的阈值选择法（Entropy)
1. 计算图像的灰度直方图，并计算对应的概率分布。概率分布可以通过灰度级别的像素数量除以总像素数量来得到。<br />2. 遍历灰度直方图中每个灰度值作为阈值，阈值将灰度小于和大于它的分为两类。分别计算两部分的灰度直方图和概率分布。分别计算两部分的熵H1和H2，并计算总熵H = H1 + H2。<br />3. 当总熵H最大时的阈值作为最终阈值。这个阈值被用来分割图像，将图像分为背景和前景两部分。<br />核心代码（二维最大熵方法）
```
def Entropy2d(P):
    h=P.shape[0]
    maxEntropy=0
    besti=0
    bestj=0
    lnp=np.log(P)
    lnp[np.isinf(lnp)]=0
    H =-np.sum(lnp*P)
    for i in range(h):
        p1=0
        hst=0
        for j in range(h):
            p1=p1+np.sum(P[0:i+1,j])
            if p1==0:continue
            hst=hst-np.sum(P[0:i+1,j]*lnp[0:i+1,j])
            Hsta=hst/p1+np.log(p1)
            Hstb=(H-hst)/(1-p1)+np.log(1-p1)
            if (Hsta+Hstb)>maxEntropy:
                maxEntropy=(Hsta+Hstb)
                besti = i
                bestj = j
    return besti,bestj
```
<a name="jz1yA"></a>
### 1.3 结果展示
<a name="Q2CDN"></a>
#### 最大类间方差的阈值选择法（OTSU）
![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703077052197-772b4436-f9cd-407a-9bf6-4a36e0c37ea2.png#averageHue=%235a5a5a&clientId=u7b53015e-bc4d-4&from=paste&height=383&id=ubba4f019&originHeight=574&originWidth=1633&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=395194&status=done&style=none&taskId=ue0172692-13d3-4e72-b347-d1c709df740&title=&width=1088.6666666666667)
<a name="haatA"></a>
#### 最大熵方法的阈值选择法（Entropy)
一维<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703076915419-42742afd-d6e8-4c31-bc92-b33869ad6e9c.png#averageHue=%23373737&clientId=u7b53015e-bc4d-4&from=paste&height=377&id=u3542fa35&originHeight=565&originWidth=1623&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=383290&status=done&style=none&taskId=u4b38d6fc-a726-4a80-8b96-ccf734c43fc&title=&width=1082)<br />二维<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703076651491-eb60649a-66c6-4216-b333-29cf198b4666.png#averageHue=%23545454&clientId=u7b53015e-bc4d-4&from=paste&height=381&id=u499a5722&originHeight=572&originWidth=1630&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=325030&status=done&style=none&taskId=u720a71f3-3ee9-4350-928f-e16a3730be7&title=&width=1086.6666666666667)
<a name="BHiaC"></a>
### 1.4 结果分析
测试集上的平均IoU与DICE结果如下，由于传统方法只根据原图片的灰度值进行分割，无法从结果标签中学习到图像分割方法，对于图片四周边界等区域也进行分割，所以平均指标很低。

| mIoU | 0.16165094723577592 |
| --- | --- |
| mDICE | 0.26151247202734823 |



<a name="IBblL"></a>
## 2 分水岭
<a name="we3H5"></a>
### 2.1 算法简介
将图像灰度图视为3维空间，每个像素的灰度值代表高度。其中的灰度值较大的像素可以看做山峰，灰度值较小的像素可以看做山谷。刚开始用水填充每个孤立的山谷(局部最小值)，当水平面上升到一定高度时，水就会溢出当前山谷，可以通过在分水岭上修大坝，从而避免山谷间的水汇集。最终这些大坝形成的线就对整个图像进行了分区，实现对图像的分割。
<a name="NPsh7"></a>
### 2.2 算法实现
1.将灰度图进行阈值分割，得到二值图，分别为前景和背景。<br />2.使用背景侵蚀前景得到一定为前景的山峰区域，使用前景膨胀得到一定为背景的山谷区域。不属于前景和背景区域为过渡区域。<br />3.对不同山谷区域进行分类，找到不同山谷区域灰度值最小的像素点，水平面从最小值开始增长，在增长的过程中，在不同山谷的水平面相交像素上设置大坝，这样就对这些邻域像素进行了分类。<br />核心代码（区域划分分水岭）
```
kernel=np.ones((7,7),np.uint8)
opening=cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=5)
# opening=cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel,iterations=2)
kernel=np.ones((3,3),np.uint8)
cv.imshow('opening',opening)
sure_bg=cv.dilate(opening,kernel,iterations=3)
sure_fg=cv.erode(opening,kernel,iterations=3)
cv.imshow('sure_bg',sure_bg)
# distance=cv.distanceTransform(opening,cv.DIST_L2,5)
# r,sure_fg=cv.threshold(distance,0.7*distance.max(),255,0)
cv.imshow('sure_fg',sure_fg)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow('unknown',unknown)
ret, markers = cv.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers+1
# 现在让所有的未知区域为0
markers[unknown==255] = 0
markers=cv.watershed(img,markers)
img[markers==-1]=[255,0,0]
```
<a name="sYham"></a>
### 2.3 结果展示
![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703080541641-329a59a2-6c40-4141-8d31-d1f4b52f8a08.png#averageHue=%233b3b3b&clientId=u7b53015e-bc4d-4&from=paste&height=257&id=uad54936d&originHeight=556&originWidth=528&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=276234&status=done&style=none&taskId=u3bcd6c17-d472-434a-9727-0772ae05035&title=&width=244)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703077883648-5c9faf97-dca6-438f-8c0f-10c92496f93a.png#averageHue=%23474747&clientId=u7b53015e-bc4d-4&from=paste&height=380&id=u89d17a60&originHeight=570&originWidth=1631&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=331069&status=done&style=none&taskId=u85586f44-ec54-43c4-91b2-4de0e88c713&title=&width=1087.3333333333333)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703080573331-dea52f4a-981a-4c37-b12b-bec42415b0e7.png#averageHue=%232b2b2b&clientId=u7b53015e-bc4d-4&from=paste&height=133&id=u93f1d7c9&originHeight=403&originWidth=776&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=249070&status=done&style=none&taskId=u0285c4fb-8513-46b3-8b2c-6f0a2a9541f&title=&width=255.3333740234375)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703078142913-79042495-a346-4476-84f4-ce1c68c44a34.png#averageHue=%231b1b1b&clientId=u7b53015e-bc4d-4&from=paste&height=287&id=u821dc5f8&originHeight=430&originWidth=2347&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=299740&status=done&style=none&taskId=u82739ad5-3a27-4641-a642-a225c065761&title=&width=1564.6666666666667)
<a name="Qtzbx"></a>
### 2.4 结果分析
测试集上的平均IoU与DICE结果如下，由于传统方法只根据原图片的灰度值进行分割，无法从结果标签中学习到图像分割方法，对于图片四周边界等区域也进行分割，所以平均指标很低。但分水岭算法由于对图片进行不同区域划分，结果要优于阈值法。

| mIoU | 0.23439532984519016 |
| --- | --- |
| mDICE | 0.3370576481534886 |



<a name="P8xFu"></a>
## 3 水平集
<a name="gtrp9"></a>
### 3.1 算法简介
 水平集分割算法是一种图像分割方法，通过演化水平集函数来实现对图像中不同对象或区域的分割。它基于水平集函数的演化，能够比较准确地捕捉图像中对象的边界。
<a name="cb327"></a>
### 3.2 算法实现
1. 初始化：确定感兴趣的区域并设置初始的水平集函数。<br />2. 定义能量函数：包括数据项和正则化项。数据项用于衡量水平集函数与图像之间的契合程度，正则化项则用于控制水平集函数的平滑性。<br />3. 演化：通过最小化能量函数来演化水平集函数，使其逐渐趋向于图像中不同对象或区域的边界。<br />4. 分割结果提取：根据演化后的水平集函数，可以得到图像的分割结果。<br />核心代码（水平集函数演化）
```
def CV(LSF, img, mu, nu, epison, dt):
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    Hea = 0.5 * (1 + (2 / math.pi) * np.arctan(LSF / epison))
    gd= np.gradient(LSF)
    gd_norm=gd/(np.sqrt(gd[0]**2 + gd[1]**2)+ 0.000001)
    Mxx, Nxx = np.gradient(gd_norm[0])
    Nyy, Myy = np.gradient(gd_norm[1])
    cur = Nxx + Nyy
    Length = nu * Drc * cur

    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu * (Lap - cur)

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))

    LSF = LSF + dt * (Length +Penalty+ CVterm)
    return LSF
```
<a name="MjADJ"></a>
### 3.3 结果展示
![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703080095731-5660970c-3c71-4ed1-99c5-ff3bd4459f2e.png#averageHue=%234c4c4c&clientId=u7b53015e-bc4d-4&from=paste&height=379&id=uab8a844f&originHeight=568&originWidth=1629&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=113662&status=done&style=none&taskId=ud0797914-32e1-43f8-8cbb-3eb4a275220&title=&width=1086)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703079888010-b58e8e9c-94af-4bba-80c0-5e6637e43dd1.png#averageHue=%23404040&clientId=u7b53015e-bc4d-4&from=paste&height=359&id=uccb6483d&originHeight=538&originWidth=1825&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=140050&status=done&style=none&taskId=ue44e74ea-bc38-44f0-8b17-5157c3953c8&title=&width=1216.6666666666667)
<a name="fsuh1"></a>
### 3.4 结果分析
测试集上的平均IoU与DICE结果如下，由于传统方法只根据原图片的灰度值进行分割，无法从结果标签中学习到图像分割方法，对于图片四周边界等区域也进行分割，所以平均指标很低。但水平集算法由于演化水平集函数来实现对图像中不同对象或区域的分割，结果要优于阈值法。

| mIoU | 0.19342593178407524 |
| --- | --- |
| mDICE | 0.2807976318308483 |



<a name="E1raP"></a>
## 4 U-Net
<a name="KBOMS"></a>
### 4.1 算法简介
Unet是一种深度学习架构，它采用编码-解码的U形结构，包括一个编码器和一个解码器。编码器对图像进行逐层的卷积和下采样，增加通道数并减少像素大小，用于提取图像特征。解码器通过上采样和卷积操作将提取的特征进行逐层解码，并和编码器同层的提取特征进行融合，最终生成与输入图像尺寸相匹配的分割结果。结合了卷积神经网络和跳跃连接，能够准确地捕捉图像中的细节信息，并具有较好的分割效果。
<a name="rcjKf"></a>
### 4.2 算法实现
1. 数据处理：处理用于训练的图像数据和相应的标签图像，进行等比例放缩并加padding，使它们均为256*256的图像。<br />2. 构建网络：定义Unet网络结构，包括卷积层、池化层和跳跃连接。<br />3. 损失函数：选择损失函数用于衡量预测分割结果与真实标签之间的差异。<br />4. 模型训练：使用训练数据对Unet模型进行训练，通过反向传播算法不断调整网络参数以最小化损失函数。<br />6. 预测和应用：利用训练好的Unet模型对新的图像进行分割预测。 <br />核心代码（网络架构）
```
class Conv(nn.Module):
    def __init__(self,C_in,C_out):
        super(Conv,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(C_in,C_out,3,1,1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.layer(x)
class DownSampling(nn.Module):
    def __init__(self,C):
        super(DownSampling,self).__init__()
        self.Down=nn.Sequential(
            nn.Conv2d(C,C,3,2,1),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.Down(x)
class UpSampling(nn.Module):
    def __init__(self,C):
        super(UpSampling,self).__init__()
        self.Up=nn.Conv2d(C,C//2,1,1)
    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.C1=Conv(1,64)

        self.D1=DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512,1024)

        self.U1=UpSampling(1024)
        self.C6=Conv(1024,512)
        self.U2=UpSampling(512)
        self.C7=Conv(512,256)
        self.U3=UpSampling(256)
        self.C8=Conv(256,128)
        self.U4=UpSampling(128)
        self.C9=Conv(128,64)

        self.Th=torch.nn.Sigmoid()
        self.pred=torch.nn.Conv2d(64,1,1,1)

    def forward(self,x):
        R1=self.C1(x)
        R2=self.C2(self.D1(R1))
        R3=self.C3(self.D2(R2))
        R4=self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))
        O1=self.C6(self.U1(Y1,R4))
        O2=self.C7(self.U2(O1,R3))
        O3=self.C8(self.U3(O2,R2))
        O4=self.C9(self.U4(O3,R1))
        return self.Th(self.pred(O4))
```
<a name="gt9Hy"></a>
### 4.3 结果展示
数据集共有562组，选取462组进行训练，剩下100组进行测试。<br />部分结果展示，左侧为输入，中间为输出，右侧为groundtruth<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703065047271-f14bd459-ec3a-42c2-8f8f-e0780587f0fc.png#averageHue=%23272727&clientId=u872fe6e8-5d1c-4&from=paste&height=173&id=u4b6af5e9&originHeight=260&originWidth=776&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=100850&status=done&style=none&taskId=u1c972abf-9f15-4469-80e9-a161ed1da0e&title=&width=517.3333333333334)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703065271361-b76d3b4e-0706-481b-9f83-2c2a1f488858.png#averageHue=%231a1a1a&clientId=u872fe6e8-5d1c-4&from=paste&height=173&id=uaf0711ab&originHeight=260&originWidth=776&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=67903&status=done&style=none&taskId=u7301579a-024f-4e22-b387-5a182059d33&title=&width=517.3333333333334)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/40483545/1703065314907-77d55bd5-ee47-4dd0-865d-8d54d9c3136a.png#averageHue=%233a3a3a&clientId=u872fe6e8-5d1c-4&from=paste&height=173&id=u9786f94a&originHeight=260&originWidth=776&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=99724&status=done&style=none&taskId=ubcb15bbe-bd5e-4437-9171-8ed9aeee62f&title=&width=517.3333333333334)
<a name="TZ8eo"></a>
### 4.4 结果分析
batchsize设为16，训练150轮结果。loss=0.0101946322247<br />测试集上的平均IoU与DICE结果如下，由于深度学习方法可以从结果标签中学习到图像分割方法，分割结果与标签结果比较，效果遥遥领先于传统方法。

| mIoU | 0.7018411391459307 |
| --- | --- |
| mDICE | 0.8107133068909971 |

<a name="lbo75"></a>
## 附
完整代码见<br />[https://github.com/1190202002/imagetest](https://github.com/1190202002/imagetest)

<a name="QOnrz"></a>
## 
<a name="Hk143"></a>
#### 
