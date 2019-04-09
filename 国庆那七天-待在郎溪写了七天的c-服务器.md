---
title: '国庆那七天:待在郎溪写了七天的c++服务器'
date: 2016-10-08 22:29:45
categories: 编程
tags: [c++,code,服务器,编程]
---

<Excerpt in index | 首页摘要> 

学习C++服务器是准备做一个游戏服务器的,类型是一个文本经营类游戏,算式游戏服务器中逻辑比较少的吧,所以准备用这个入门一下,然后鞋这些代码也是在框架下去写的,先学着怎么在框架下去写逻辑然后才是去学习着怎么去写框架吧,学习路线循序渐进慢慢来.

<!-- more -->

<The rest of contents | 余下全文>

# 读取配置表和属性表

## 作用

假如我们现在写的是阴阳师的服务,那么属性表就是最基础是东西,比如玩家的id,玩家的level,玩家的名字之类的.而配置表呢就是玩家的一些成长属性了,比如玩家现在是14级,那么玩家现在的一些基础属性就可以去配置表里面对应的14级的数据去读取,然后配置到客户端中,这两个表是用xml写的,传输给客户端的是protobuf,这个后面讲.

## XML的结构

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ExchangeList> 
    <Exchange
        ExchangeLevel="1"
        CityLevel="2"
        MaxPrincipal="1000"
        MaxProfit="100"
        ProfitMargin="100"
        BuildingCost="120009,1|120017,23"
        UpgradeCondition="2,2,3"
        UpgradeCost="10,2222"
    />
    <Exchange
        ExchangeLevel="2"
        CityLevel="3"
        MaxPrincipal="2000"
        MaxProfit="200"
        ProfitMargin="200"
        BuildingCost="0"
        UpgradeCondition="2,2,4"
        UpgradeCost="10,2223"
    />
</ExchangeList> 
```

读写的代码在下面

## 代码

这个代码是放在轮子下得,所以这个可能在别的下面运行不起来

```c++
TiXmlElement * root = exchange_configure.RootElement();//声明一个XML的对象指针
	TASSERT(root, "wtf");//检查一下死否为空

	TiXmlElement * env = root->FirstChildElement("Exchange");//找到第一个节点
	RowIndex index = NULL;
	while (env) {
		TASSERT(env && env->Attribute("ExchangeLevel"), "wtf");//检查这个节点有没有值
		//_env_exchange_configure = env->Attribute("share");
		/*插入数据*/
		//RowIndex index = NULL;
		index = g_exchange_table_config->AddRowKeyInt64(tools::StringAsInt64(env->Attribute("ExchangeLevel")));//按照key去存取数据
		g_exchange_table_config->SetDataInt64(index, game::table::exchangetable::column_city_level_int64, tools::StringAsInt64(env->Attribute("CityLevel")));
		g_exchange_table_config->SetDataInt64(index, game::table::exchangetable::column_max_principal_int64, tools::StringAsInt64(env->Attribute("MaxPrincipal")));
		g_exchange_table_config->SetDataInt64(index, game::table::exchangetable::column_max_profit_int64, tools::StringAsInt64(env->Attribute("MaxProfit")));
		g_exchange_table_config->SetDataInt64(index, game::table::exchangetable::column_profit_margin_int64, tools::StringAsInt64(env->Attribute("ProfitMargin")));
		g_exchange_table_config->SetDataString(index, game::table::exchangetable::column_building_cost_string, env->Attribute("BuildingCost"));
		g_exchange_table_config->SetDataString(index, game::table::exchangetable::column_upgrade_condition_string, env->Attribute("UpgradeCondition"));
		g_exchange_table_config->SetDataString(index, game::table::exchangetable::column_upgrade_cost_string, env->Attribute("UpgradeCost"));

		//index->GetDataInt64();
		env = env->NextSiblingElement("Exchange");//循环取下一个节点
	}
```

# 数据存储过程

## 作用

因为逻辑服务器和数据服务器是分开的,所以我们会有一个数据存储的过程(毕竟是一个慢节奏的游戏,所以对于数据的反应并不是特别的要求),我们是用c++去请求php的请求去进行的数据的存储然后又php去调用数据库的存储过程去进行数据的存储.