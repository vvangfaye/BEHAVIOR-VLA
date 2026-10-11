"""Pure, bounded evidence checks embedded into the read-only audit worker."""
import math


def capture_last_teacher_pngs(attempt):
 """Optional current-camera evidence; never a training/success input."""
 import base64
 from contextlib import ExitStack
 import stat
 result={'diagnostic_pngs':[],'diagnostic_pngs_capture':{'state':'unavailable'}}
 try:
  if attempt.is_symlink() or attempt.parent.is_symlink():raise ValueError('Linked attempt directory')
  with ExitStack() as opened:
   def directory(name,parent=None):
    fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
    opened.callback(os.close,fd)
    return fd
   attempt_fd=directory(attempt);teacher_fd=directory('teacher',attempt_fd)
   candidates=[]
   for name in os.listdir(teacher_fd):
    match=re.fullmatch(r'step-(\d{8})(?:-decision-(\d{4}))?',name)
    if match is None:continue
    step=int(match[1]);index=int(match[2]) if match[2] is not None else -1
    if step>10000 or index>=32:raise ValueError('Diagnostic step exceeds collection scope')
    candidates.append((step,index,name))
    if len(candidates)>32:raise ValueError('Diagnostic step count exceeds collection budget')
   if not candidates:raise FileNotFoundError('No teacher observation exists')
   step,index,name=max(candidates);snapshot_fd=directory(name,teacher_fd)
   images=[]
   for camera in range(3):
    fd=os.open('camera-'+str(camera)+'.png',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=snapshot_fd)
    with os.fdopen(fd,'rb') as stream:
     before=os.fstat(stream.fileno())
     if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=512*1024:
      raise ValueError('Diagnostic PNG size or file type differs')
     raw=stream.read(512*1024+1);after=os.fstat(stream.fileno())
    if (len(raw)!=before.st_size or before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns
        or not raw.startswith(b'\x89PNG\r\n\x1a\n')):raise ValueError('Diagnostic PNG changed or has no PNG signature')
    images.append({'camera':camera,'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),
                   'base64':base64.b64encode(raw).decode('ascii')})
   # Bound the encoded report field too, not only the sum of the source bytes.
   if len(json.dumps({'diagnostic_pngs':images},indent=2).encode())>1536*1024:
    raise ValueError('Encoded PNG report exceeds diagnostic budget')
   result={'diagnostic_pngs':images,'diagnostic_pngs_capture':{'state':'complete','step':step,
            'decision_index':None if index<0 else index,'directory':name}}
 except Exception as error:
  result['diagnostic_pngs_capture']['error_type']=type(error).__name__
 return result


def correction_diagnostics(value):
 if not isinstance(value,dict):return None
 result={}
 if value.get('arm') in {'left','right'}:result['arm']=value['arm']
 for key in ('max_joint_delta_rad','joint_delta_limit_rad','joint_lower_violation_rad','joint_upper_violation_rad',
             'linearized_position_residual_m','linearized_position_limit_m',
             'linearized_orientation_residual_rad','linearized_orientation_limit_rad'):
  item=value.get(key)
  if type(item) in (int,float) and math.isfinite(item) and 0<=item<1e6:result[key]=item
 return result


def decision_location(path,request):
 step=request.get('step');index=request.get('decision_index')
 if type(step) is not int or not 0<=step<=10000:raise ValueError('Teacher step is invalid')
 if path.parent.parent.name=='step-'+format(step,'08d') and 'decision_index' not in request:
  return {'layout':'legacy'}
 if (type(index) is not int or not 0<=index<32
     or path.parent.parent.name!='step-'+format(step,'08d')+'-decision-'+format(index,'04d')):
  raise ValueError('Teacher decision directory/index differs from request')
 return {'layout':'indexed','decision_index':index}


def decision_order(entries):
 valid=[row for row in entries if row.get('decision_valid')]
 if not valid:return []
 layouts={row['layout'] for row in valid}
 if len(layouts)!=1:return ['mixed_teacher_directory_layouts']
 steps=[row['step'] for row in valid]
 if steps!=sorted(steps):return ['teacher_steps_not_monotonic']
 if layouts=={'indexed'}:
  if [row['decision_index'] for row in valid]!=list(range(len(valid))):return ['teacher_decision_indices_not_contiguous']
 elif len(set(steps))!=len(steps):return ['teacher_step_inconsistent']
 return []


def audit_collection_mode(value):
 if value not in {'teacher_conditioned','native_with_corrections'}:raise ValueError('Unknown collection mode')
 return value


def action_summary(path,max_steps=4500,collection_mode='teacher_conditioned'):
 native=audit_collection_mode(collection_mode)=='native_with_corrections'
 counts=Counter();rows=[];issues=[];segments=[];recent=[]
 if not path.is_file():return {'present':False,'rows':0,'sources':{},'segments':[],'issues':['action_log_missing']}
 with path.open(encoding='utf-8') as stream:
  for number,line in enumerate(stream):
   if number>=max_steps:issues.append('action_budget_exceeded');break
   try:
    row=json.loads(line)
    if type(row.get('step')) is not int or row['step']!=number:raise ValueError('Non-contiguous action steps')
    source=row.get('source')
    expected_sources={'teacher_cartesian_correction','policy_native_subtask' if native else 'policy_conditioned_on_teacher_subtask'}
    if source not in expected_sources:raise ValueError('Unknown action source')
    binding={}
    if 'decision_index' in row or 'request_id' in row:
     unbound=(native and source=='policy_native_subtask' and 'decision_index' in row and 'request_id' in row
              and row['decision_index'] is None and row['request_id'] is None)
     if not unbound and (type(row.get('decision_index')) is not int or not 0<=row['decision_index']<32
         or not isinstance(row.get('request_id'),str) or re.fullmatch('[0-9a-f]{64}',row['request_id']) is None):
      raise ValueError('Invalid executed decision binding')
     binding={key:row[key] for key in ('decision_index','request_id')}
    elif native:raise ValueError('Native action must declare its optional decision binding')
    if native:
     label=row.get('subtask');label_source=row.get('subtask_source')
     if (not isinstance(label,str) or not label or label!=' '.join(label.split()) or any(ord(c)<32 for c in label)
         or label_source not in ({'provided'} if source=='teacher_cartesian_correction' else {'generated','cached'})):
      raise ValueError('Native action lacks an actual policy label or provenance')
     binding.update(subtask_sha256=hashlib.sha256(label.encode()).hexdigest(),subtask_source=label_source)
    counts[source]+=1;rows.append(row['step'])
    recent.append({'step':row['step'],'source':source,**binding,'diagnostics':correction_diagnostics(row.get('diagnostics'))});recent=recent[-8:]
    if (segments and segments[-1]['source']==source and segments[-1]['end']==row['step']
        and {key:segments[-1][key] for key in ('decision_index','request_id','subtask_sha256','subtask_source') if key in segments[-1]}==binding):segments[-1]['end']+=1
    else:segments.append({'start':row['step'],'end':row['step']+1,'source':source,**binding})
   except (ValueError,TypeError,AttributeError):issues.append('invalid_action_row_'+str(number))
 return {'present':True,'rows':len(rows),'sources':dict(counts),'segments':segments,'recent_actions':recent,'issues':issues,'sha256':sha(path)}


def correction_outcome(value):
 if value is None:return None
 if not isinstance(value,dict) or value.get('status') not in {'rejected','executed'}:raise ValueError('Invalid correction outcome')
 rejected=value['status']=='rejected'
 required={'status','step','decision_index','request_id','executed','diagnostics'}
 if rejected:required|={'error_type','reason_code'}
 if (set(value)!=required or type(value.get('step')) is not int or not 0<=value['step']<=10000
     or type(value.get('decision_index')) is not int or not 0<=value['decision_index']<32
     or not isinstance(value.get('request_id'),str) or re.fullmatch('[0-9a-f]{64}',value['request_id']) is None
     or value.get('executed') is not (not rejected)):
  raise ValueError('Correction outcome identity or execution flag differs')
 if rejected and (value['error_type']!='RecoverableCorrectionError' or value['reason_code'] not in {'ik_joint_bounds','ik_residual'}):
  raise ValueError('Unknown recoverable rejection')
 diagnostic=correction_diagnostics(value.get('diagnostics'))
 if not diagnostic or diagnostic!=value['diagnostics']:raise ValueError('Correction diagnostics have unsupported/nonfinite values')
 return {**value,'diagnostics':diagnostic}


def validate_rejection(value,decision):
 row=correction_outcome(value)
 if (row['status']!='rejected' or decision.get('action')!='correct'
     or row['request_id']!=decision['request_id'] or row['decision_index']!=decision.get('decision_index')
     or not decision['step']<=row['step']<decision['step']+decision['correction_steps']):
  raise ValueError('Rejection is not a pending candidate from this correction')
 correction=decision['correction'];diag=row['diagnostics']
 base={'arm','max_joint_delta_rad','joint_delta_limit_rad','joint_lower_violation_rad','joint_upper_violation_rad'}
 residual={'linearized_position_residual_m','linearized_position_limit_m','linearized_orientation_residual_rad','linearized_orientation_limit_rad'}
 if (set(diag)!=(base|residual if row['reason_code']=='ik_residual' else base)
     or diag['arm']!=correction['arm'] or diag['joint_delta_limit_rad']!=0.15):
  raise ValueError('Rejection diagnostics differ from bounded solver contract')
 joint_failure=diag['max_joint_delta_rad']>0.15+1e-6 or diag['joint_lower_violation_rad']>0 or diag['joint_upper_violation_rad']>0
 if row['reason_code']=='ik_joint_bounds':
  if not joint_failure:raise ValueError('No joint bound was rejected')
 else:
  pos=max(0.002,math.sqrt(sum(v*v for v in correction['delta_position_m']))*0.25)
  ori=max(0.01,math.sqrt(sum(v*v for v in correction['delta_rotation_axis_angle_rad']))*0.25)
  if (joint_failure or not math.isclose(diag['linearized_position_limit_m'],pos,rel_tol=1e-12,abs_tol=1e-12)
      or not math.isclose(diag['linearized_orientation_limit_rad'],ori,rel_tol=1e-12,abs_tol=1e-12)
      or not (diag['linearized_position_residual_m']>pos or diag['linearized_orientation_residual_rad']>ori)):
   raise ValueError('No unchanged residual boundary was rejected')
 return row


def audit_rejections(path,teachers,actions):
 entries=[];issues=[];by_id={row['request_id']:row for row in teachers['entries'] if row.get('decision_valid')}
 indexed=any(row.get('layout')=='indexed' for row in teachers['entries'])
 if not path.is_file():
  return {'present':False,'rows':0,'entries':[],'issues':['correction_rejections_missing'] if indexed else []}
 seen=set()
 with path.open(encoding='utf-8') as stream:
  for number,line in enumerate(stream):
   if number>=32:issues.append('correction_rejection_budget_exceeded');break
   try:
    value=json.loads(line);decision=by_id.get(value.get('request_id'))
    if decision is None:raise ValueError('Rejected request is unverified')
    row=validate_rejection(value,decision)
    if row['decision_index'] in seen or row['step']>actions['rows']:raise ValueError('Duplicate or future rejection')
    if entries and (row['decision_index']<=entries[-1]['decision_index'] or row['step']<entries[-1]['step']):raise ValueError('Rejection order differs')
    seen.add(row['decision_index']);entries.append(row)
   except (ValueError,TypeError,AttributeError,KeyError):issues.append('invalid_correction_rejection_'+str(number))
 return {'present':True,'rows':len(entries),'entries':entries,'issues':issues,'sha256':sha(path)}


def execution_consistency(teachers,actions,trajectory,rejections=None,collection_mode='teacher_conditioned',teacher_interval=256):
 native=audit_collection_mode(collection_mode)=='native_with_corrections'
 if native and (type(teacher_interval) is not int or teacher_interval<=0):raise ValueError('Native teacher interval is invalid')
 rejections=rejections or {'entries':[]};issues=[]
 ordered=[row for row in teachers['entries'] if row.get('decision_valid')]
 rejected={row['decision_index']:row for row in rejections['entries']}
 count=trajectory['steps'] if trajectory.get('valid_complete_trajectory') else actions['rows']
 if any(row['step']>count for row in ordered):issues.append('teacher_step_inconsistent')
 by_id={row['request_id']:row for row in ordered}
 executed={index:row for row in actions['segments'] for index in range(row['start'],row['end'])}
 indexed=any(row.get('layout')=='indexed' for row in ordered)
 if native:
  if any(row.get('layout')!='indexed' for row in ordered):issues.append('native_decision_requires_indexed_layout')
  if ordered and ordered[0]['step']!=teacher_interval:issues.append('native_first_teacher_step_differs')
 if indexed:
  for current,following in zip(ordered,ordered[1:]):
   event=rejected.get(current.get('decision_index'))
   if following['step']==current['step'] or (current['action']=='correct' and following['step']<current['step']+current['correction_steps']):
    if event is None or event['step']!=following['step']:issues.append('replan_without_matching_rejection')
   if event is not None and following['step']!=event['step']:issues.append('replan_advanced_past_rejection')
   if native:
    due=event['step'] if event is not None else current['step']+(current['correction_steps'] if current['action']=='correct' else teacher_interval)
    if current['action']=='abort' or following['step']!=due:issues.append('native_teacher_schedule_differs')
  for decision in ordered:
   outcome=decision.get('last_correction_outcome')
   if outcome is None:continue
   if outcome['decision_index']>=decision['decision_index'] or outcome['step']>decision['step']:
    issues.append('correction_feedback_order_differs');continue
   if outcome['status']=='rejected':
    if rejected.get(outcome['decision_index'])!=outcome:issues.append('correction_feedback_missing_rejection')
   else:
    action=executed.get(outcome['step'],{})
    if (outcome['step']>=decision['step'] or action.get('source')!='teacher_cartesian_correction'
        or action.get('request_id')!=outcome['request_id'] or action.get('decision_index')!=outcome['decision_index']):
     issues.append('correction_feedback_execution_differs')
  for event in rejections['entries']:
   following=next((row for row in ordered if row.get('decision_index')==event['decision_index']+1),None)
   if following is not None and following.get('last_correction_outcome')!=event:issues.append('rejection_feedback_not_delivered')
 labels={index:row['subtask_sha256'] for row in trajectory.get('segments',[]) for index in range(row['start'],row['end'])}
 current=None;cursor=0
 for index in range(count):
  while cursor<len(ordered) and ordered[cursor]['step']<=index:current=ordered[cursor];cursor+=1
  action=executed.get(index,{})
  if native and current is None:
   if (index>=teacher_interval or action.get('source')!='policy_native_subtask'
       or action.get('decision_index') is not None or action.get('request_id') is not None):
    issues.append('native_initial_execution_differs')
   if trajectory.get('valid_complete_trajectory') and labels.get(index)!=action.get('subtask_sha256'):
    issues.append('executed_subtask_differs_from_action_log')
   continue
  if current is None or current['action']=='abort':issues.append('executed_without_active_teacher_decision');continue
  if indexed and (action.get('decision_index')!=current.get('decision_index') or action.get('request_id')!=current['request_id']):
   issues.append('executed_decision_binding_differs')
  event=rejected.get(current.get('decision_index'))
  if event is not None and index>=event['step']:issues.append('rejected_candidate_was_executed')
  expected='teacher_cartesian_correction' if current['action']=='correct' and index<current['step']+current['correction_steps'] else ('policy_native_subtask' if native else 'policy_conditioned_on_teacher_subtask')
  if action.get('source')!=expected:issues.append('executed_action_source_differs_from_teacher')
  if native:
   due=current['step']+(current['correction_steps'] if current['action']=='correct' else teacher_interval)
   if index>=due:issues.append('native_teacher_interval_missed')
   if trajectory.get('valid_complete_trajectory') and labels.get(index)!=action.get('subtask_sha256'):
    issues.append('executed_subtask_differs_from_action_log')
  if (trajectory.get('valid_complete_trajectory') and (not native or expected=='teacher_cartesian_correction')
      and labels.get(index)!=current['subtask_sha256']):issues.append('executed_subtask_differs_from_teacher')
 # Also reject a stale earlier request bound to an action after its rejected step.
 for index,action in executed.items():
  event=rejected.get(action.get('decision_index'))
  if event is not None and action.get('request_id')==event['request_id'] and index>=event['step']:
   issues.append('rejected_candidate_was_executed')
 return sorted(set(issues))
