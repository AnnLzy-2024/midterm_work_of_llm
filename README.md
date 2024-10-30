# midterm_work_of_llm
midterm work

硬件条件有限，只训练了极小体量模型，数据集选取了140K的GPT生成的英语短篇童话故事，即目录中的novel.txt。

模型大小约600M，无法上传，可至百度网盘获取：百度网盘分享的文件：model.pth 链接：https://pan.baidu.com/s/1KiGN8EEIEoCYnGJX5TwI_w  提取码：6gw0 

直接在本地训练更快。

1、cd 对应目录

2、pip install -r requirements.txt

3、python gpt_train.py   生成并保存model.pth文件

4、python gpt_generate.py  生成语句，输入可至代码line 78进行修改

本模型非常简陋，无法保证所有生成内容可通顺理解，实验后挑选出了10句符合条件的输出语句进行展示，控制台截图在word文档中

1、	One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom

2、	Once upon a time, there was a little flower named Bloom. Bloom lived in a big garden with many other flowers.

3、	Lucy asked the flame, "Can you let me fly away so I can be happy again?"

4、	The boy arrived at the room to show her silver necklace in the next day

5-8、She got very excited and started to crawl into the hole. Once she got in she saw the most amazing thing: thousands of brightly coloured stones and sparkly crystals! She was so excited that she couldn't move. Suddenly, a small mouse said: "Why hello there little turtle! What are you doing here?"

9-10、Timmy said, "Fluffy, the word is 'repeat'. Can you say 'repeat'?" Fluffy looked at Timmy and said, "Meow." Timmy laughed and said, "No, Fluffy, say 'repeat'."
