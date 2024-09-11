#  核心速览

## 研究背景

1. **研究问题**：这篇文章要解决的问题是点击率（CTR）预测中的特征交互建模。具体来说，现有的深度交叉网络（DCN）及其衍生模型在计算成本和性能之间取得了有效平衡，但存在四个主要局限性：显式特征交互方法的性能通常弱于隐式深度神经网络（DNN）；许多模型在增加特征交互阶数时未能有效过滤噪声；大多数模型的融合方法无法为其不同子网络提供适当的监督信号；尽管大多数模型声称能够捕捉高阶特征交互，但它们通常通过DNN隐式且不可解释地实现这一点，这限制了模型预测的可信度。

2. **研究难点**：该问题的研究难点包括：如何在不使用DNN的情况下提高显式特征交互的性能；如何在增加特征交互阶数的同时有效过滤噪声；如何为不同子网络提供适当的监督信号；如何提高模型的可解释性。

3. 相关工作

    ：该问题的研究相关工作有：传统的线性模型如LR和FM，以及深度学习模型如DNN、PNN、Wide & Deep、DeepFM、DCNv1、DCNv2、DIN、FiGNN等。这些模型在捕捉特征交互方面有不同的方法和优缺点。

    ![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/bc2bb5ed72ceb3ebfc8311798c77a36110aca3f9.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=5a97f75cc2099bd623f3ad50ca12f4124a2cbde5)

## 研究方法

这篇论文提出了下一代深度交叉网络（DCNv3），用于解决CTR预测中的特征交互建模问题。具体来说，

1. **线性交叉网络（LCN）**：用于低阶（浅层）显式特征交互，采用线性增长的交互方法。其递归公式如下：

cl=Wl⋅xl−1+bl*c**l*=*W**l*⋅*x**l*−1+*b**l*xl=xl−1⊙[cl∣∣Mask(cl)∣]+xl−1*x**l*=*x**l*−1⊙[*c**l*∣∣Mask(*c**l*)∣]+*x**l*−1

其中，cl*c**l*表示第l*l*层的交叉向量，Wl*W**l*和bl*b**l*分别是可学习的权重矩阵和偏置向量，xl−1*x**l*−1是第l−1*l*−1层的特征交互，MaskMask表示自掩码操作。

![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/b733c7e0c6b979bbc07517b7b345902b0dbedf0c.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=932f7937cbf49f91529cabb4416190dcbd28685f)



1. **指数交叉网络（ECN）**：用于高阶（深层）显式特征交互，采用指数增长的交互方法。其递归公式如下：

ce=We⋅xe−1−1+be*c**e*=*W**e*⋅*x**e*−1−1+*b**e*xe=xe−1⊙[ce∣∣Mask(ce)∣]+xe−1*x**e*=*x**e*−1⊙[*c**e*∣∣Mask(*c**e*)∣]+*x**e*−1

其中，ce*c**e*表示第e*e*层的交叉向量，We*W**e*和be*b**e*分别是可学习的权重矩阵和偏置向量，xe−1*x**e*−1是第e−1*e*−1层的特征交互，MaskMask表示自掩码操作。

3. **自掩码操作（Self-Mask）**：用于过滤噪声并减少交叉网络中的参数数量。其计算过程如下：

Mask(cl)=cl⊙max⁡(0,LN(cl))Mask(*c**l*)=*c**l*⊙max(0,LN(*c**l*))LN(cl)=g⋅Norm(cl)+bLN(*c**l*)=*g*⋅Norm(*c**l*)+*b*

其中，LNLN表示层归一化，g*g*和b*b*是参数，cl*c**l*是交叉向量。

![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/ea4dfc48210ba5573c714874369e791be230f3fb.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=893bdfa14237f6f1b5a9a63efcb1224432bdf4a7)



1. **融合层**：通过LCN和ECN的预测结果进行融合，避免使用DNN：

y^=12(We⋅xe+be)+12(Wl⋅xl+bl)*y*^=21(W*e*⋅*x**e*+*b**e*)+21(W*l*⋅*x**l*+*b**l*)

其中，y^*y*^是最终预测结果，We*W**e*和Wl*W**l*是可学习的权重，be*b**e*和bl*b**l*是偏置。

5. **Tri-BCE损失**：提出了一种简单而有效的多损失计算方法，称为Tri-BCE，以提供适当的监督信号：

L=−∑i=1n(yilog⁡(y^i)+(1−yi)log⁡(1−y^i))*L*=−*i*=1∑*n*(*y**i*log(*y*^*i*)+(1−*y**i*)log(1−*y*^*i*))Ltri=L+wec⋅Lec+wlc⋅Llc*L*tri=*L*+*w**ec*⋅*L**ec*+*w**l**c*⋅*L**l**c*

其中，yi*y**i*是真实标签，y^i*y*^*i*是预测结果，Lec*L**ec*和Llc*L**l**c*分别是ECN和LCN的损失，wec*w**ec*和wlc*w**l**c*是自适应权重。

## 实验设计

1. **数据集**：在六个CTR预测数据集上进行实验，包括Avazu、Criteo、ML-1M、KDD12、iPinYou和KKBox。
2. **数据预处理**：对数据进行预处理，包括时间戳字段的转换、数值特征字段的离散化处理以及不常见分类特征的替换。
3. **评估指标**：使用Logloss和AUC两个常用指标来比较模型性能。AUC衡量正例排在负例之前的概率，Logloss衡量模型拟合数据的能力。
4. **基线模型**：与一些最先进的模型进行比较，包括仅显式特征交互的模型和集成隐式特征交互的模型。
5. **实现细节**：使用PyTorch实现所有模型，采用Adam优化器，设置默认学习率为0.001，嵌入维度根据数据集不同而有所不同，批量大小根据数据集不同而有所不同，使用早停法防止过拟合。

## 结果与分析

1. **整体性能（RQ1）**：DCNv3在所有六个数据集上均表现出色，平均AUC提升0.21%，平均Logloss降低0.11%，均超过统计显著阈值0.1%。ECN在AUC方面表现最佳，而DCNv3在Logloss优化方面表现更好。

    ![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/e0eb9958f50e3bafafb1472027089294ceaf0a91.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=e488bd7710f38513f38e6e36f8245a72665e60c8)

    

2. **效率比较（RQ2）**：显式CTR模型通常参数较少，运行时间较短。DCNv3和ECN是最参数高效的模型，分别在Avazu和Criteo数据集上达到SOTA性能，同时保持较高的运行效率。

    ![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/94041483b2468f980ca55df896dbe2af88f83e5d.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=3ab9fb1a427ce428499a9fd6997aac7017b98a34)

    

3. **可解释性和噪声过滤能力（RQ3）**：通过动态交叉向量和静态权重向量分析了模型的预测过程，发现自掩码操作有效地过滤了噪声，提高了模型的可解释性。

    ![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/8aeb16b91764751ad3e144b5cd9d81f1f7b4daa7.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=28f7799f4eec0073ed31dc4c2736bf8564a4ee31)

    

4. **消融研究（RQ4）**：去除Tri-BCE损失、去除层归一化后的模型性能有所下降，表明这些组件对模型性能的必要性。LCN和ECN在不同网络深度下的性能也进行了实验，发现ECN在高阶特征交互方面表现更优。

    ![img](https://pcg-pdf-1258344706.cos.ap-nanjing.myqcloud.com/f5b10cd6ee668ca99f94a0ccd37d078c8b1f653c.jpeg?q-sign-algorithm=sha1&q-ak=AKIDPegYHoiRZifN0VwunU1MaeaLwJVW75c4&q-sign-time=1726061344%3B1728653944&q-key-time=1726061344%3B1728653944&q-header-list=&q-url-param-list=&q-signature=a300a4c93d5f916ff01eeacc59390692560fd375)

    

## 总体结论

本文提出了DCNv3，一种新的显式特征交互建模方法，通过LCN和ECN分别捕捉低阶和高阶特征交互，并使用自掩码操作过滤噪声。Tri-BCE损失为不同子网络提供了适当的监督信号。实验结果表明，DCNv3在多个CTR基准测试中取得了第一名的成绩，打破了传统CTR模型集成隐式特征交互以提高性能的惯例。

# 论文评价

## 优点与创新

1. **首次实现仅使用显式特征交互建模**：本文首次在不集成DNN的情况下实现了令人惊讶的性能，这可能颠覆了以往CTR预测文献中的范式，激发了进一步重新审视和创新特征交互建模的潜力。
2. **引入新的深度交叉方法**：提出了一种新的指数增长的深度交叉方法，称为Deep Crossing，以显式捕捉高阶特征交互，同时将之前的交叉方法归类为浅层交叉。
3. **自掩码操作**：设计了一种自掩码操作来过滤噪声并减少交叉网络中的参数数量，使其减半。
4. **多损失计算方法**：在融合层中，提出了一种简单而有效的多损失计算和权衡方法，称为Tri-BCE，以确保不同子网络获得适当的监督信号。
5. **实验验证**：在六个数据集上的综合实验验证了DCNv3的有效性、效率和可解释性。基于实验结果，模型在多个CTR预测基准测试中获得了第一名。

## 不足与反思

1. **模型复杂度分析**：尽管本文提出的ECN和DCNv3在参数效率和运行时间上表现出色，但由于使用了Tri-BCE损失，DCNv3的损失计算时间复杂度是其他模型的三倍，这可能会影响实际应用中的推理速度。
2. **下一步工作**：未来的研究可以进一步优化模型的计算效率，例如通过改进损失函数的计算方式或引入更高效的优化算法来减少训练时间。

# 关键问题及回答

**问题1：DCNv3中的线性交叉网络（LCN）和指数交叉网络（ECN）在特征交互方法上有何不同？它们分别适用于什么样的特征交互场景？**

线性交叉网络（LCN）用于低阶（浅层）显式特征交互，采用线性增长的交互方法。其递归公式如下：

cl=Wl⋅xl−1+bl*c**l*=*W**l*⋅*x**l*−1+*b**l*xl=xl−1⊙[cl∣∣Mask(cl)∣]+xl−1*x**l*=*x**l*−1⊙[*c**l*∣∣Mask(*c**l*)∣]+*x**l*−1

LCN适用于捕捉低阶特征交互，能够在有限层数内有效地进行特征交互，适合用于小规模或中等规模的数据集。

指数交叉网络（ECN）用于高阶（深层）显式特征交互，采用指数增长的交互方法。其递归公式如下：

ce=We⋅xe−1−1+be*c**e*=*W**e*⋅*x**e*−1−1+*b**e*xe=xe−1⊙[ce∣∣Mask(ce)∣]+xe−1*x**e*=*x**e*−1⊙[*c**e*∣∣Mask(*c**e*)∣]+*x**e*−1

ECN适用于捕捉高阶特征交互，能够在较少的层数内实现特征交互的指数增长，适合用于大规模数据集和需要高效特征交互的场景。

**问题2：DCNv3中的自掩码操作（Self-Mask）是如何工作的？它在特征交互过程中起到了什么作用？**

自掩码操作（Self-Mask）用于过滤噪声并减少交叉网络中的参数数量。其计算过程如下：

Mask(cl)=cl⊙max⁡(0,LN(cl))Mask(*c**l*)=*c**l*⊙max(0,LN(*c**l*))LN(cl)=g⋅Norm(cl)+bLN(*c**l*)=*g*⋅Norm(*c**l*)+*b*

其中，LNLN表示层归一化，g*g*和b*b*是参数，cl*c**l*是交叉向量。

在特征交互过程中，自掩码操作通过将交叉向量的每个元素与层归一化后的结果进行逐元素乘积，从而过滤掉噪声信息。具体来说，层归一化确保了掩码操作后交叉向量中包含大约50%的零值，这些零值对应于无用的噪声特征交互，从而减少了模型的参数数量和计算复杂度，同时提高了模型的泛化能力和预测精度。

**问题3：DCNv3中的Tri-BCE损失函数是如何设计的？它如何为不同子网络提供适当的监督信号？**

Tri-BCE损失函数用于提供适当的监督信号。其计算过程如下：

L=−∑i=1n(yilog⁡(y^i)+(1−yi)log⁡(1−y^i))*L*=−*i*=1∑*n*(*y**i*log(*y*^*i*)+(1−*y**i*)log(1−*y*^*i*))Ltri=L+wec⋅Lecn+wlc⋅Llcn*L**t**r**i*=*L*+*w**ec*⋅*L**ec**n*+*w**l**c*⋅*L**l**c**n*

其中，yi*y**i*是真实标签，y^i*y*^*i*是预测结果，wec*w**ec*和wlc*w**l**c*是自适应权重。

Tri-BCE损失函数通过为ECN和LCN分别设置自适应权重wec*w**ec*和wlc*w**l**c*，使得这两个子网络在训练过程中获得不同的监督信号。具体来说，Tri-BCE损失函数的主要损失L*L*是二元交叉熵损失，辅助损失Lecn*L**ec**n*和Llcn*L**l**c**n*分别针对ECN和LCN的预测结果。通过联合训练这两个子网络，Tri-BCE损失函数能够动态调整权重，确保每个子网络根据其在总体损失中的贡献获得合适的监督信号，从而提高模型的整体性能。