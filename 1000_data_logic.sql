--取数逻辑：1300人：1300人
1，在ris中获得2674个病人的patient-local-id
select distinct t1.ris_no 
from tjris.exam_master t1 
inner join tjris.exam_report t2 
        on t1.ris_no = t2.ris_no 
where t1.modality like '%CT%' 
and   t1.exam_itemsstr like '%胸部%' 
--and   t1.exam_itemsstr like '%成人%’ 
and   t1.exam_itemsstr like '%平扫%' 
--and t1.patient_source = '门诊' 
--(t2.impression like '%癌%' or t2.impression like '%肿瘤%' ) or (t2.description like '%癌%' or t2.description like '%肿瘤%') 
and 
( 
 (t2.impression like '%考虑肿瘤性病变%' or t2.description like '%考虑肿瘤性病变%') 
or
(t2.impression like '%考虑为肿瘤性病变%' or t2.description like '%考虑为肿瘤性病变%') 
) 
and (t2.impression not  like '%术后%' and t2.impression not like '%术后%' ) 
and t2.lastsavetime >  to_date('2014-01-01','YYYY-MM-DD') 
and t2.lastsavetime <  to_date('2016-05-01','YYYY-MM-DD') 

2，将这2674人导入pacs库，然后利用pacs库的条件限制这个人有1.25mm序列，得到1300个有病且有1.25mm图像的病人（正常的片子需要计算，不正常的按照正常的量截取即可）
select distinct t1.internal_euid
from synapse.patient t1 
inner join synapse.study t2 
on t1.id = t2.patient_uid 
inner join synapse.series t3 
on t2.id = t3.study_uid 
inner join   synapse.image t4 
on t3.id = t4.series_uid 
inner join   synapse.image_version t5 
on t4.id = t5.image_uid 
inner join   synapse.storage t6 
on t5.storage_uid  = t6.id 
inner join   synapse.tx_ct_chest_abnormal_2674 t7 
on t1.internal_euid = t7.patient_local_id 
where  t3.image_count >=192 and t3.image_count <= 336
这个结果集大于1300，需要截取一部分平衡正常人的数据

3，用1300个有1.25mm影像的有病的人的patient-local-id 寻找其对应的40W+图片，为1300人建表速度较 in（结果集）快,然后导出csv文件
select distinct t1.internal_euid , t6.unc_path,t5.filename 
from synapse.patient t1 
inner join synapse.study t2 
on t1.id = t2.patient_uid 
inner join synapse.series t3 
on t2.id = t3.study_uid 
inner join synapse.image t4 
on t3.id = t4.series_uid 
inner join synapse.image_version t5 
on t4.id = t5.image_uid 
inner join synapse.storage t6 
on t5.storage_uid = t6.id 
inner join synapse.tx_ct_chest_abnormal_1300_125 t7  
on t1.internal_euid = t7.patient_local_id 
where  t3.image_count >=192 and t3.image_count <= 336

4，将第三步的40W图片导入ris库（在ris中建表，因为需要从ris表获取大量汉字报告），再和ris中的exam_master，exam_report 关联，得到最终需要的数据的格式，其中有人有多套，但是不会有小于192的
insert into tjris.tx_ct_abnormal_path_40w_0615
select distinct t1.ris_no,t1.patient_local_id,t2.description,t2.impression,t3.unc_path,t3.filename ,1 as flag 
from tjris.exam_master t1 
inner join tjris.exam_report t2 
on t1.ris_no = t2.ris_no 
inner join tjris.TX_CT_CHEST_ABNORMAL_PATH_1300 t3 
on t1.patient_local_id = t3.internal_euid 
inner join 
( 
select t3.internal_euid,count(t3.filename) as img_cnt 
from tjris.TX_CT_CHEST_ABNORMAL_PATH_1300 t3 
group by t3.internal_euid 
order by img_cnt 
) t4 
on t1.patient_local_id = t4.internal_euid 
where t4.img_cnt <=336 and t4.img_cnt >=192
order by
t1.patient_local_id,
t3.unc_path,
cast(substr(t3.filename,1,8) as int),
cast(substr(t3.filename,instr(t3.filename,'.',1,10)+1,2) as int),
cast(substr(t3.filename,instr(t3.filename,'.',1,11)+1,3) as int)



5，经过第4步去重，筛选，40w变成32w（1200+人），还需要根据每个人一套片子的总数，间隔取数，此时把32w放到一张表，然后进行取模的操作
-- 没病的  62262 ---63255
insert into tjris.tx_ct_normal_final_7w_0615
select * 
from 
(select 
 t.RIS_NO     , 
 t.PATIENT_LOCAL_ID , 
 t.DESCRIPTION   , 
 t.IMPRESSION   , 
 t.UNC_PATH    , 
 case when 
 (case when t2.cnt >= 192 and t2.cnt <240 
   then mod(cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int),4) 
   when t2.cnt >= 240 and t2.cnt <288 
   then mod(cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int),5) 
   when t2.cnt >= 288 and t2.cnt <=329 
   then mod(cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int),6) else 0 end ) = 1 
   then t.filename 
      else 'pass' end as filename    , 
 t.FLAG  
from tjris.tx_ct_normal_path_40w_0615 t 
inner join 
(
select t1.PATIENT_LOCAL_ID, 
    count(t1.filename) as cnt 
from tjris.tx_ct_normal_path_40w_0615 t1 
group by t1.patient_local_id 
) t2 
on t.patient_local_id = t2.patient_local_id 
where t.filename not in ('pass') 
) t11 
where t11.filename not in ('pass') 
--有病的
--  63459  -- 65202 
insert into tjris.tx_ct_abnormal_final_7w_0615
select * 
from 
(select 
 t.RIS_NO     , 
 t.PATIENT_LOCAL_ID , 
 t.DESCRIPTION   , 
 t.IMPRESSION   , 
 t.UNC_PATH    , 
 case when 
 (case when t2.cnt >= 192 and t2.cnt <240 
   then mod(cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int),4) 
   when t2.cnt >= 240 and t2.cnt <288 
   then mod(cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int),5) 
   when t2.cnt >= 288 and t2.cnt <=329 
   then mod(cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int),6) else 0 end ) = 1 
   then t.filename 
      else 'pass' end as filename    , 
 t.FLAG  
from tjris.tx_ct_abnormal_path_40w_0615 t 
inner join 
(
select t1.PATIENT_LOCAL_ID, 
    count(t1.filename) as cnt 
from tjris.tx_ct_abnormal_path_40w_0615 t1 
group by t1.patient_local_id 
) t2 
on t.patient_local_id = t2.patient_local_id 
where t.filename not in ('pass') 
) t11 
where t11.filename not in ('pass') 

7，经过第六步的处理，有病没病各有7w多图片，每个人需要48张，用rownumber获取每套片子的前48张，filename升序排列
select * from
(
select RIS_NO  ,
 PATIENT_LOCAL_ID ,
 DESCRIPTION ,
 IMPRESSION ,
 UNC_PATH ,
 FILENAME,
  FLAG
,row_number() over(partition by t.patient_local_id order by cast(substr(t.filename,instr(t.filename,'.',-1,1)+1,8) as int) ) as rn
from tjris.tx_ct_abnormal_final_7w_0615 t
) t11
where t11.rn <=48



































