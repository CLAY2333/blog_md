---

title: Google Protocol Buffer简单的使用
date: 2016-09-26 11:54:21
categories: 数据格式
tags: [protobuf,code,编程]
---

<Excerpt in index | 首页摘要> 

Google Protocol Buffer( 简称 Protobuf) 是 Google 公司内部的混合语言数据标准，目前已经正在使用的有超过 48,162 种报文格式定义和超过 12,183 个 .proto 文件。他们用于 RPC 系统和持续数据存储系统。

Protocol Buffers 是一种轻便高效的结构化数据存储格式，可以用于结构化数据串行化，或者说序列化。它很适合做数据存储或 RPC 数据交换格式。可用于通讯协议、数据存储等领域的语言无关、平台无关、可扩展的序列化结构数据格式。目前提供了 C++、Java、Python 三种语言的 API.

<!-- more -->

<The rest of contents | 余下全文>

# 什么是数据格式

> 数据格式（data format）是数据保存在文件或记录中的编排格式。可为数值、字符或二进制数等形式。由数据类型及数据长度来描述。
>

介绍一下现在比较常用的数据格式

## XML

XML(可扩展标记性语言),XML在web方面的开发应用比较广泛.它是目前比较流行的数据格式.

```xml
<?xml version="1.0" encoding="utf-8" ?>
<country>
  <name>中国</name>
  <province>
    <name>黑龙江</name>
    <citys>
      <city>哈尔滨</city>
      <city>大庆</city>
    </citys>  　　
  </province>
  <province>
    <name>广东</name>
    <citys>
      <city>广州</city>
      <city>深圳</city>
      <city>珠海</city>
    </citys> 　　
  </province>
  <province>
    <name>台湾</name>
    <citys>
      　<city>台北</city>
      　<city>高雄</city>
    </citys>　
  </province>
  <province>
    <name>新疆</name>
    <citys>
      <city>乌鲁木齐</city>
    </citys>
  </province>
</country>
```



下面看看XML的优点缺点:

> 
>
> <1>.XML的优点
> 　　A.格式统一，符合标准；
> 　　B.容易与其他系统进行远程交互，数据共享比较方便。
> <2>.XML的缺点
> 　　A.XML文件庞大，文件格式复杂，传输占带宽；
> 　　B.服务器端和客户端都需要花费大量代码来解析XML，导致服务器端和客户端代码变得异常复杂且不易维护；
> 　　C.客户端不同浏览器之间解析XML的方式不一致，需要重复编写很多代码；
> 　　D.服务器端和客户端解析XML花费较多的资源和时间。

简单的优缺点就可以看出来,XML是一个用途比较广泛但是使用复杂的一个数据格式,使用和学习起来都不是很简单

## JSON

JSON(JavaScript Object Notation) 是一种轻量级的数据交换格式,他的语法习惯类似于c语言的习惯.

```json
{
            name: "中国",
            provinces: [
            { name: "黑龙江", citys: { city: ["哈尔滨", "大庆"]} },
            { name: "广东", citys: { city: ["广州", "深圳", "珠海"]} },
            { name: "台湾", citys: { city: ["台北", "高雄"]} },
            { name: "新疆", citys: { city: ["乌鲁木齐"]} }
            ]
}
```

这个是就很简单而且轻量级,下面也看看它的优缺点



> JSON的优点：
>
> A.数据格式比较简单，易于读写，格式都是压缩的，占用带宽小；
>
> B.易于解析，客户端JavaScript可以简单的通过eval()进行JSON数据的读取；
>
> C.支持多种语言，包括ActionScript, C, C#, ColdFusion, Java, JavaScript, Perl, PHP, Python, Ruby等服务器端语言，便于服务器端的解析；
>
> D.在PHP世界，已经有PHP-JSON和JSON-PHP出现了，偏于PHP序列化后的程序直接调用，PHP服务器端的对象、数组等能直接生成JSON格式，便于客户端的访问提取；
>
> E.因为JSON格式能直接为服务器端代码使用，大大简化了服务器端和客户端的代码开发量，且完成任务不变，并且易于维护。
>
> JSON的缺点
>
> A.没有XML格式这么推广的深入人心和喜用广泛，没有XML那么通用性；
>
> B.JSON格式目前在Web Service中推广还属于初级阶段。

# Google Protocol Buffer

Protobuf 是一个开源项目,是谷歌公司开发出来的,本来是XML生存得好好地但是Google公司为啥不用呢,因为XML的性能不够好,Google就不缺这些造轮子的人才,所以Protobuf就这样的产生了.

## 一个简单的例子

### 书写.proto文件

首先我们需要编写一个proto文件,定义我们需要处理的结构化数据,在protobuf的术语中,结构化数据被称为message.proto文件并非常类似于java或者语言的数据定义.

```properties
package lm; 
 message helloworld 
 { 
    required int32     id = 1;  // ID 
    required string    str = 2;  // str 
    optional int32     opt = 3;  //optional field 
 }
```

一个比较好的习惯是认真对待 proto 文件的文件名。比如将命名规则定于如下： packageName.MessageName.proto

在上例中，package 名字叫做 lm，定义了一个消息 helloworld，该消息有三个成员，类型为 int32 的 id，另一个为类型为 string 的成员 str。opt 是一个可选的成员，即消息中可以不包含该成员。

### 编译.proto文件

写好 proto 文件之后就可以用 Protobuf 编译器将该文件编译成目标语言了。本例中我们将使用 C++。

假设您的 proto 文件存放在 $SRC_DIR 下面，您也想把生成的文件放在同一个目录下，则可以使用如下命令：

```
 protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto
```

命令将生成两个文件：

lm.helloworld.pb.h ， 定义了 C++ 类的头文件

lm.helloworld.pb.cc ， C++ 类的实现文件

在生成的头文件中，定义了一个 C++ 类 helloworld，后面的 Writer 和 Reader 将使用这个类来对消息进行操作。诸如对消息的成员进行赋值，将消息序列化等等都有相应的方法。

### 编写write和reader文件

wrtie:

```c++
 #include "lm.helloworld.pb.h"
…

 int main(void) 
 { 
  
  lm::helloworld msg1; 
  msg1.set_id(101); 
  msg1.set_str(“hello”); 
        
  if (!msg1.SerializeToOstream(&output)) { 
      cerr << "Failed to write msg." << endl; 
      return -1; 
  }         
  return 0; 
 }
```

reader:

```c++
#include "lm.helloworld.pb.h" 
…
 void ListMsg(const lm::helloworld & msg) { 
  cout << msg.id() << endl; 
  cout << msg.str() << endl; 
 } 
 
 int main(int argc, char* argv[]) { 

  lm::helloworld msg1; 
 
  { 
    if (!msg1.ParseFromIstream(&input)) { 
      cerr << "Failed to parse address book." << endl; 
      return -1; 
    } 
  }
  ListMsg(msg1); 
  … 
 }
```

运行结果:

```
 >writer 
 >reader 
 101 
 Hello
```

可以看出来protobuf在对数据处理中是特别简单的,速度上也是很快毕竟google大厂优化.在需要改动与一些数据的时候我们可以简单的加几句话就可以增加这个protobuf的数据结构.

## protobuf的优缺点

>### Protobuf 的优点
>
>Protobuf 有如 XML，不过它更小、更快、也更简单。你可以定义自己的数据结构，然后使用代码生成器生成的代码来读写这个数据结构。你甚至可以在无需重新部署程序的情况下更新数据结构。只需使用 Protobuf 对数据结构进行一次描述，即可利用各种不同语言或从各种不同数据流中对你的结构化数据轻松读写。
>
>它有一个非常棒的特性，即“向后”兼容性好，人们不必破坏已部署的、依靠“老”数据格式的程序就可以对数据结构进行升级。这样您的程序就可以不必担心因为消息结构的改变而造成的大规模的代码重构或者迁移的问题。因为添加新的消息中的 field 并不会引起已经发布的程序的任何改变。
>
>Protobuf 语义更清晰，无需类似 XML 解析器的东西（因为 Protobuf 编译器会将 .proto 文件编译生成对应的数据访问类以对 Protobuf 数据进行序列化、反序列化操作）。
>
>使用 Protobuf 无需学习复杂的文档对象模型，Protobuf 的编程模式比较友好，简单易学，同时它拥有良好的文档和示例，对于喜欢简单事物的人们而言，Protobuf 比其他的技术更加有吸引力。
>
>### Protobuf 的不足
>
>Protbuf 与 XML 相比也有不足之处。它功能简单，无法用来表示复杂的概念。
>
>XML 已经成为多种行业标准的编写工具，Protobuf 只是 Google 公司内部使用的工具，在通用性上还差很多。
>
>由于文本并不适合用来描述数据结构，所以 Protobuf 也不适合用来对基于文本的标记文档（如 HTML）建模。另外，由于 XML 具有某种程度上的自解释性，它可以被人直接读取编辑，在这一点上 Protobuf 不行，它以二进制的方式存储，除非你有 .proto 定义，否则你没法直接读出 Protobuf 的任何内容

简单的先写到这里,后面可以试着看看复杂一点的protobuf.

