--1:有且仅心脏有病.
--2:有且仅肺部有病.
--3:完全正常.
--4:异常且不包含前三个集合。四个集合互相不重复,只记录过滤数据规则。不包括完整sql语句
--health people
select distinct t1.ris_no
       ,t1.patient_local_id
       ,t2.description
       ,t2.impression
from tjris.exam_master t1
inner join tjris.exam_report t2
on t1.ris_no = t2.ris_no
where modality = 'CR'
and t1.patient_source = '门诊'
and t1.exam_itemsstr = '数字拍片-胸部正位(成人)'
and (t2.impression  = '所见心肺膈未见明显异常。'
    or t2.impression = '双肺、心、膈未见明显异常。'
    or t2.impression = '心肺膈未见明显异常。'
    or t2.impression = '心、肺、膈未见明显异常。'
    or t2.impression = '两肺、心、膈未见明显异常。'
    or t2.impression = '双肺、心、膈影未见明显异常。'
    )

--only heart problem
select distinct t1.ris_no
       ,t1.patient_local_id
       ,t2.description
       ,t2.impression
from tjris.exam_master t1
inner join tjris.exam_report t2
      on t1.ris_no = t2.ris_no
where modality = 'CR'
and t1.patient_source = '门诊'
and t1.exam_itemsstr = '数字拍片-胸部正位(成人)'
and (t2.impression like '%心%' or t2.impression like '%主动脉结%' )
and t2.impression not like '%肺%'
and t2.impression not like '%积液%'
and t2.impression not like '%术后%'
and t2.impression not like '%骨折%'
and t2.impression not like '%肋膈角%'
and t2.impression not like '%无明显异常%'
and t2.impression not like '%心、膈未见异常%'
and t2.impression not like '%心膈未见明显异常%'
and t2.impression not like '%心、膈未见明显异常%'

--only lung problem
select distinct t1.ris_no
       ,t1.patient_local_id
       ,t2.description
       ,t2.impression
from tjris.exam_master t1
inner join tjris.exam_report t2
      on t1.ris_no = t2.ris_no
left join  tjris.tx_cr_only_heart_sick_0811 t3
on t1.patient_local_id = t3.patient_local_id
and t3.patient_local_id is null
where modality = 'CR'
and t1.patient_source = '门诊'
and t1.exam_itemsstr = '数字拍片-胸部正位(成人)'
and ( t2.impression like '%肺动脉%' 
      or t2.impression like '%肋膈角钝%'  
      or t2.impression like '%胸膜增厚%' 
      or t2.impression like '%肺感染%')
and t2.impression not like '%心%'
and t2.impression not like '%主动脉%'
and t2.impression not like '%术后%'
and t2.impression not like '%积液%'
and t2.impression not like '%骨折%'

-- sick people not contain (only heart sick and only lung sick)
select  t1.ris_no
       ,t1.patient_local_id
       ,t2.description
       ,t2.impression
from tjris.exam_master t1
inner join tjris.exam_report t2
on t1.ris_no = t2.ris_no
where modality = 'CR'
and t1.patient_source = '门诊'
and t1.exam_itemsstr = '数字拍片-胸部正位(成人)'
and (t2.impression like '%肋膈角钝%' or t2.impression like '%心影增大%'
or t2.impression like '%主动脉结突出%' or t2.impression like '%胸膜增厚%' or t2.impression like '%肺感染%')
and t1.patient_local_id not in (select t11.patient_local_id from tjris.tx_cr_only_heart_sick_0811 t11)
and t1.patient_local_id not in (select t12.patient_local_id from tjris.tx_cr_only_lung_sick_0811 t12)