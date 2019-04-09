---
title: PHP学习之路(1)
date: 2016-09-18 12:47:51
categories: 编程
tags: [php,code,编程]
---

<Excerpt in index | 首页摘要> 

学习PHP的原因是因为实验室的项目的服务器是用LAMP的平台弄得,但是在之前只有一下C/C++的编程经验,所以我也是从新手开始学,现在就是慢慢把我学习的路写出来,希望对你有帮助.



<!-- more -->

<The rest of contents | 余下全文>

# PHP的难度

其实我觉得php真的是一门比较简单的语言了,它不像c一样有什么复杂的变量转换,更不会像C++一样有复杂的面向对象,毕竟PHP的面向的小中型的服务器开发,在这点上他的特性就够用了,所以如果学习服务器是怎么工作原理是什么我觉得php很适合入门.

# PHP语法

php的语法和c是很像的,大家可以在`[w3school](http://www.w3school.com.cn/php/index.asp)上面去学习一下基础语法,这里我贴一些语法.

## 简单的输出

### echo

php中最简单的输出,能够输出一个以上的字符串

```php
<?php
  echo 1;
  echo "Hello world";
  $test="hello word"
  echo $test;
  $test=1;
  echo $test;
  echo "welcome", " to", " php";
?>
```

### print

大家对这个都很熟悉了,和echo的区别就是只能输出一个字符串,始终返回1;

```php
print "welcome to php";
print "welcome","to","php";(错的)
```

### dump

这个输出是输出结构,比如array数组或者json.

```php
$cars=array("hellp","word","23333");
dump($cars);
```

上面就是给大家看看吧,详细的语法还是去w3school去看吧

## 一个简单的登录注册

下面就直接写点实际的东西吧,先上代码

### 登录代码

```php
<?php
    $mysql_server_name="localhost"; //数据库服务器名称
    $mysql_username="root"; // 连接数据库用户名
    $mysql_password=""; // 连接数据库密码
    $mysql_database=""; // 数据库的名字
    // 连接到数据库
    $con = mysql_connect($mysql_server_name,$mysql_username,$mysql_password);
    mysql_select_db($mysql_database, $con);
     // 从表中提取信息的sql语句
    $username=$_GET['username'];
    $passwd=$_GET['passwd'];
    echo $username;
    echo $passwd;
    // 执行sql查询
    $result = mysql_query("select id from account where username='{$username}' and passwd='{$passwd}'");
    $row = mysql_fetch_array($result);
    if($row[id]==NULL){
        echo "login fail!";
    }
    else{
        echo "login success!";
    }
?>
```

估计会报出很多的warning因为这个数据库连接的函数有点老了,先比较好用的是PDO,不过这不影响我们学习.

数据库用的是mysql,链接代码就不说了是固定的.想了解PDO的话左转前往百度,送你一个传送门[baidu](http://www.baidu.com).

```php
$test_GET=$_GET['test_GET'];//GET传参
$test_POST=$_POST['test_POST'];//POST传参
```



这两个是接受传过来的参数,php主要有两种方式,一个是GET一个是POST.

GET:

> - GET 请求可被缓存
> - GET 请求保留在浏览器历史记录中
> - GET 请求可被收藏为书签
> - GET 请求不应在处理敏感数据时使用
> - GET 请求有长度限制
> - GET 请求只应当用于取回数据

POST:

>- POST 请求不会被缓存
>- POST 请求不会保留在浏览器历史记录中
>- POST 不能被收藏为书签
>- POST 请求对数据长度没有要求

简单的说GET传值是在你的url里面发送的,也就是你请求网页的时候那传很长的网址

> https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=1&tn=baidu&wd=markdown&oq=markdown%20%E8%A1%A8%E6%A0%BC%20%E8%AF%AD%E6%B3%95&rsv_pq=bb87f6fa000884f6&rsv_t=e867ztgMucWZladUc5sgHLke41tVIrZsE2QYqbexZiVS%2FmN0V2HIUNugzQs&rqlang=cn&rsv_enter=1&inputT=213&rsv_sug3=31&rsv_sug1=38&rsv_sug7=100&rsv_sug2=0&rsv_sug4=213&rsv_sug=2&bs=markdown%20%E8%A1%A8%E6%A0%BC%20%E8%AF%AD%E6%B3%95

这是一个百度搜索的url,这里面就包含了我请求的网络GET请求,比较我百度的makedown,我用的编码是utf-8等等,这些都是你可以很明显看到的.但是POST传值的话就是放在了http请求里面了类似于这样:

```http
POST /test/demo_form.asp HTTP/1.1
Host: w3schools.com
name1=value1&name2=value2
```



这个信息里面包含了接口位置/test/dome_form.asp,http协议1.1,变量名.但是这些参数需要你用专用浏览器或者抓包的工具拿这个http请求,所以看起来GET比较简单方便但是容易泄露一些信息,POST比较复杂但是安全高一些.使用的时候看情况使用.

```php
    $result = mysql_query("select id from account where username='{$username}' and passwd='{$passwd}'");
```

简单的赋值,这个内容是sql语句,在w3school里面也可以学习的没什么好讲的.

```php
    $row = mysql_fetch_array($result);
    if($row[id]==NULL){
        echo "login fail!";
    }
    else{
        echo "login success!";
    }
```

最后这个就是运行我们的sql代码,获取结果然后判断,是否找到一个用户名和密码都匹配的用户,是的话输出成功,不是的话就输出失败,那么这个简单的登录就弄完了.

### 注册代码



```php
<?php
       
            $mysql_server_name="localhost"; //数据库服务器名称
            $mysql_username="root"; // 连接数据库用户名
            $mysql_password=""; // 连接数据库密码
            $mysql_database=""; // 数据库的名字
            $con = mysql_connect($mysql_server_name,$mysql_username,$mysql_password);
            mysql_select_db($mysql_database, $con);
            $username=$_GET['username'];
            $passwd=$_GET['passwd'];
            $result = mysql_query("SELECT id FROM account where account.email='{$username}'");
            $row = mysql_fetch_array($result);
            if($row['id']!=NULL){
                echo "register fail!";
            }else{
                  $result = mysql_query("INSERT INTO account (username,passwd) values ('{$username}','{$passwd}')");
                  $row = mysql_fetch_array($result);
            }
            mysql_close($con);
     
```

最基本数据库的操作是没什么变化的

```php
$result = mysql_query("SELECT id FROM account where account.email='{$username}'");
$row = mysql_fetch_array($result);
if($row['id']!=NULL){
                echo "register fail!";
            }else{
                  $result = mysql_query("INSERT INTO account (username,passwd) values ('{$username}','{$passwd}')");
                  $row = mysql_fetch_array($result);
            }
```

注册的时候登录名肯定是要独一无二的,所以要用查询这个注册的用户名是否在数据库中存在,如果存在的话呢么id是存在的,就会返回失败,如果不存在就可以进行数据的插入这个时候就插入成功了.

### 结束语

其实PHP是属于很简单的服务器的入门语言了,先学习这门语言去了解服务器的运行机制,然后再去学习C++或者java服务器的时候就会少一些弯路.

