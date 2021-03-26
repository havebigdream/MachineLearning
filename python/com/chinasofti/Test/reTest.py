"""
 -*- coding:utf-8 -*-
@Time : 2020/8/12 1:18
@Author: 面向对象写Bug
@File : author.py 
@content：
"""
import re
str1 = str({
    "status": 0,
    "message": "query ok",
    "request_id": "e6ad9216-dfc4-11ea-b51a-3ce1a17a3425",
    "result": {
        "location": {
            "lat": 22.11613,
            "lng": 113.31
        },
        "address": "广东省珠海市金湾区",
        "formatted_addresses": {
            "recommend": "金湾区红旗镇兴发围",
            "rough": "金湾区红旗镇兴发围"
        },
        "address_component": {
            "nation": "中国",
            "province": "广东省",
            "city": "珠海市",
            "district": "金湾区",
            "street": "",
            "street_number": ""
        },
        "ad_info": {
            "nation_code": "156",
            "adcode": "440404",
            "city_code": "156440400",
            "name": "中国,广东省,珠海市,金湾区",
            "location": {
                "lat": 22.03134,
                "lng": 113.5
            },
            "nation": "中国",
            "province": "广东省",
            "city": "珠海市",
            "district": "金湾区"
        },
        "address_reference": {
            "town": {
                "id": "440404103",
                "title": "红旗镇",
                "location": {
                    "lat": 22.14728,
                    "lng": 113.35054
                },
                "_distance": 0,
                "_dir_desc": "内"
            },
            "landmark_l2": {
                "id": "13029442793667317179",
                "title": "兴发围",
                "location": {
                    "lat": 22.11479,
                    "lng": 113.30658
                },
                "_distance": 382.8,
                "_dir_desc": "东北"
            }
        }
    }
})

result1 = re.findall('lng.*?(\d+\.\d+)',str1)[0]
result2 = re.findall('lat.*?(\d+\.\d+)',str1)[0]
result3  = re.findall('address(.*?),',str1)[0]
result3=result3.split(": ")[1]

print(result1,result2,result3)