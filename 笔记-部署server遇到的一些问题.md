---
title: 笔记-部署server遇到的一些问题
date: 2016-12-16 00:07:55
categories: 笔记
tags: [笔记,linux,server]
---

<Excerpt in index | 首页摘要> 

今天总算是配好了整个要上线的服务器,算一算这也是我配着服务器第三次了,还能出这么错,我觉得吧这些错误都记下来.为了方便之后万一还能遇到呢,百度太不靠谱了23333.

<!-- more -->

<The rest of contents | 余下全文>

# 2016/12/16 第一次

## apache方面

### rewrite重定向

其实就是书写一个正则表达式,识别固定的域名来实现跳转,步骤如下

1. 编辑你的httpd.conf

   ```
   vim /etc/httpd/conf/httpd.conf
   ```

   这是一般的默认位置

2. 找到  LoadModule Rewrite_module modules/mod_Rewrite.so 把前面的#去掉

3. 将以下代码插入到最后即可

   ```
           RewriteEngine On
           RewriteRule ^/tpl/(.*)$ /tpl/$1 [L,NC]
           RewriteRule ^/static/(.*)$ /static/$1 [L,NC]
           RewriteRule ^/(.*)$ /index.php/$1 [L,NC]
           RewriteRule ^/(.*)$ /index.php/$1 [L,NC]
           RewriteRule ^/(.*)$ /index.php/$1 [L,NC]
           RewriteRule ^/(.*)$ /index.php/$1 [L,NC]
   ```

rewrite可以配合virtualhost使用,用来单服务器多个项目运行,其他的一些简单的地址更改就不写了.

## mysql

### 错误1130

* 错误实体:ERROR 1130: Host xxx.xxx.xxx.xxx  is not allowed to connect to this MySQL server
* 原因:链接的Host并不在user表的用户中.
* 解决方案: 在sql中的mysql数据库中的user表存着用户的信息,默认只有localhost和127.0.0.1的host可以访问,要么你讲user表中的某一个用户的host改为%或者你的IP就可以链接.或者新建一个用户host设置为%或者你的IP.

### 错误2003

* 错误实体:ERROR 2003 :Can't connect to Mysql server on xxx.xxx.xxx.xxx

* 原因:一般是防火墙屏蔽了mysql的3306端口

* 解决:开启即可

  ​

### PDO

假如代码使用了PDO,记得给安装扩展的PDO,在php上就应该装饰php-level,便于扩展还有phpize也是需要安装的.

### mysql.sock

有时候在运行mysql的时候你发现sock并没有在你程序指定的地方,这个时候我们需要更改mysql的conf文件去修改.sock文件的位置

```
[mysqld]
datadir=/var/lib/mysql
socket=/tmp/mysql.sock
#生成的位置
user=mysql
# Disabling symbolic-links is recommended to prevent assorted security risks
symbolic-links=0

[mysqld_safe]
log-error=/var/log/mysqld.log
pid-file=/var/run/mysqld/mysqld.pid

[client]
port            = 3306
socket          = /tmp/mysql.sock
#客户端机运行的位置
```

记得上面的所有东西弄完都需要重启!!重启!!重启!!!

## Redis

redis是一个超烦的东西,每次配置的时候都要弄一阵(图是别人的,自己弄得没想到截图)

* ```
  wget http://download.redis.io/redis-stable.tar.gz
  ```

  ![img](http://img.blog.csdn.net/20150802220338361)

* ```
  tar –zxvf redis-stable.tar.gz
  ```

  ![img](http://img.blog.csdn.net/20150804115342779)

* ```
  cd redis-stable
  make
  ```

  如果make的时候提示不识别,就安装gcc

  如果提示 couldn’t execute tcl : no such file or dicrectory,就安装tcl

  如果提示 in file include from ......   就执行make clean 再 make

  ![img](http://img.blog.csdn.net/20150804095708032)

* ```
  make install
  ```

  ![img](http://img.blog.csdn.net/20150804095750436)

* 基本的就安装好了,接下来你需要在你自定义的目录下创建好,data,log,run三个文件夹

* 再讲解压包下的redis.conf文件拷贝到/etc/redis下,再打开redis.conf开始配置

* ```
  #查找一下属性自定义你的redis服务
  port 6373
  #你可以修改这个端口来变你要执行再那个端口上
  #XX为你自定义的文件目录
  pidfile /XX/run/redis/pid
  dir /XX/data
  logfile /XX/log/redis.log
  daemonize no 
  #no改为yes 这个不改的话你会发现data没数据,但是redis在跑,那是redis在background运行
  ```

  保存,用这个conf启动redis   redis-server /etc/redis/redis.conf

  你可以用redis-cli测试

