Е■
═г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.12v2.3.0-54-gfcc4b966f18▄═
Э
 critic_q_1_hidden1_common/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" critic_q_1_hidden1_common/kernel
Ц
4critic_q_1_hidden1_common/kernel/Read/ReadVariableOpReadVariableOp critic_q_1_hidden1_common/kernel*
_output_shapes
:	А*
dtype0
Х
critic_q_1_hidden1_common/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name critic_q_1_hidden1_common/bias
О
2critic_q_1_hidden1_common/bias/Read/ReadVariableOpReadVariableOpcritic_q_1_hidden1_common/bias*
_output_shapes	
:А*
dtype0
Ю
 critic_q_1_hidden2_common/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" critic_q_1_hidden2_common/kernel
Ч
4critic_q_1_hidden2_common/kernel/Read/ReadVariableOpReadVariableOp critic_q_1_hidden2_common/kernel* 
_output_shapes
:
АА*
dtype0
Х
critic_q_1_hidden2_common/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name critic_q_1_hidden2_common/bias
О
2critic_q_1_hidden2_common/bias/Read/ReadVariableOpReadVariableOpcritic_q_1_hidden2_common/bias*
_output_shapes	
:А*
dtype0
Н
critic_q_1_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_namecritic_q_1_output/kernel
Ж
,critic_q_1_output/kernel/Read/ReadVariableOpReadVariableOpcritic_q_1_output/kernel*
_output_shapes
:	А*
dtype0
Д
critic_q_1_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecritic_q_1_output/bias
}
*critic_q_1_output/bias/Read/ReadVariableOpReadVariableOpcritic_q_1_output/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Р
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╦
value┴B╛ B╖
г
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
#_self_saveable_object_factories

signatures
	regularization_losses

trainable_variables
	variables
	keras_api
%
#_self_saveable_object_factories
%
#_self_saveable_object_factories
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
Н

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
Н

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
 	variables
!	keras_api
Н

"kernel
#bias
#$_self_saveable_object_factories
%regularization_losses
&trainable_variables
'	variables
(	keras_api
 
 
 
*
0
1
2
3
"4
#5
*
0
1
2
3
"4
#5
н
)non_trainable_variables
	regularization_losses
*metrics
+layer_metrics

,layers

trainable_variables
	variables
-layer_regularization_losses
 
 
 
 
 
 
н
.non_trainable_variables
/metrics
regularization_losses
0layer_metrics

1layers
trainable_variables
	variables
2layer_regularization_losses
lj
VARIABLE_VALUE critic_q_1_hidden1_common/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEcritic_q_1_hidden1_common/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
н
3non_trainable_variables
4metrics
regularization_losses
5layer_metrics

6layers
trainable_variables
	variables
7layer_regularization_losses
lj
VARIABLE_VALUE critic_q_1_hidden2_common/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEcritic_q_1_hidden2_common/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
н
8non_trainable_variables
9metrics
regularization_losses
:layer_metrics

;layers
trainable_variables
 	variables
<layer_regularization_losses
db
VARIABLE_VALUEcritic_q_1_output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEcritic_q_1_output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

"0
#1

"0
#1
н
=non_trainable_variables
>metrics
%regularization_losses
?layer_metrics

@layers
&trainable_variables
'	variables
Alayer_regularization_losses
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Л
(serving_default_critic_q_1_input_actionsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
К
'serving_default_critic_q_1_input_statesPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
┤
StatefulPartitionedCallStatefulPartitionedCall(serving_default_critic_q_1_input_actions'serving_default_critic_q_1_input_states critic_q_1_hidden1_common/kernelcritic_q_1_hidden1_common/bias critic_q_1_hidden2_common/kernelcritic_q_1_hidden2_common/biascritic_q_1_output/kernelcritic_q_1_output/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8В *.
f)R'
%__inference_signature_wrapper_1991275
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╘
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4critic_q_1_hidden1_common/kernel/Read/ReadVariableOp2critic_q_1_hidden1_common/bias/Read/ReadVariableOp4critic_q_1_hidden2_common/kernel/Read/ReadVariableOp2critic_q_1_hidden2_common/bias/Read/ReadVariableOp,critic_q_1_output/kernel/Read/ReadVariableOp*critic_q_1_output/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *)
f$R"
 __inference__traced_save_1991479
╫
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename critic_q_1_hidden1_common/kernelcritic_q_1_hidden1_common/bias critic_q_1_hidden2_common/kernelcritic_q_1_hidden2_common/biascritic_q_1_output/kernelcritic_q_1_output/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *,
f'R%
#__inference__traced_restore_1991507ўЩ
─
╛
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_1991409

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
ш
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991240

inputs
inputs_1%
!critic_q_1_hidden1_common_1991224%
!critic_q_1_hidden1_common_1991226%
!critic_q_1_hidden2_common_1991229%
!critic_q_1_hidden2_common_1991231
critic_q_1_output_1991234
critic_q_1_output_1991236
identityИв1critic_q_1_hidden1_common/StatefulPartitionedCallв1critic_q_1_hidden2_common/StatefulPartitionedCallв)critic_q_1_output/StatefulPartitionedCallФ
(critic_q_1_concatenation/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *^
fYRW
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_19910652*
(critic_q_1_concatenation/PartitionedCallЬ
1critic_q_1_hidden1_common/StatefulPartitionedCallStatefulPartitionedCall1critic_q_1_concatenation/PartitionedCall:output:0!critic_q_1_hidden1_common_1991224!critic_q_1_hidden1_common_1991226*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_199108523
1critic_q_1_hidden1_common/StatefulPartitionedCallе
1critic_q_1_hidden2_common/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden1_common/StatefulPartitionedCall:output:0!critic_q_1_hidden2_common_1991229!critic_q_1_hidden2_common_1991231*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_199111223
1critic_q_1_hidden2_common/StatefulPartitionedCall№
)critic_q_1_output/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden2_common/StatefulPartitionedCall:output:0critic_q_1_output_1991234critic_q_1_output_1991236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *W
fRRP
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_19911382+
)critic_q_1_output/StatefulPartitionedCallЪ
IdentityIdentity2critic_q_1_output/StatefulPartitionedCall:output:02^critic_q_1_hidden1_common/StatefulPartitionedCall2^critic_q_1_hidden2_common/StatefulPartitionedCall*^critic_q_1_output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::2f
1critic_q_1_hidden1_common/StatefulPartitionedCall1critic_q_1_hidden1_common/StatefulPartitionedCall2f
1critic_q_1_hidden2_common/StatefulPartitionedCall1critic_q_1_hidden2_common/StatefulPartitionedCall2V
)critic_q_1_output/StatefulPartitionedCall)critic_q_1_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
╜
f
:__inference_critic_q_1_concatenation_layer_call_fn_1991378
inputs_0
inputs_1
identityф
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *^
fYRW
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_19910652
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
┌
╢
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_1991428

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┌
╢
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_1991138

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─
╛
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_1991112

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒	
Є
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991216
critic_q_1_input_states
critic_q_1_input_actions
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallcritic_q_1_input_statescritic_q_1_input_actionsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8В *V
fQRO
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_19912012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:         
1
_user_specified_namecritic_q_1_input_states:a]
'
_output_shapes
:         
2
_user_specified_namecritic_q_1_input_actions
°
╙
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991365
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8В *V
fQRO
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_19912402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
√ 
у
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991329
inputs_0
inputs_1<
8critic_q_1_hidden1_common_matmul_readvariableop_resource=
9critic_q_1_hidden1_common_biasadd_readvariableop_resource<
8critic_q_1_hidden2_common_matmul_readvariableop_resource=
9critic_q_1_hidden2_common_biasadd_readvariableop_resource4
0critic_q_1_output_matmul_readvariableop_resource5
1critic_q_1_output_biasadd_readvariableop_resource
identityИО
$critic_q_1_concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$critic_q_1_concatenation/concat/axis╠
critic_q_1_concatenation/concatConcatV2inputs_0inputs_1-critic_q_1_concatenation/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2!
critic_q_1_concatenation/concat▄
/critic_q_1_hidden1_common/MatMul/ReadVariableOpReadVariableOp8critic_q_1_hidden1_common_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype021
/critic_q_1_hidden1_common/MatMul/ReadVariableOpф
 critic_q_1_hidden1_common/MatMulMatMul(critic_q_1_concatenation/concat:output:07critic_q_1_hidden1_common/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 critic_q_1_hidden1_common/MatMul█
0critic_q_1_hidden1_common/BiasAdd/ReadVariableOpReadVariableOp9critic_q_1_hidden1_common_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0critic_q_1_hidden1_common/BiasAdd/ReadVariableOpъ
!critic_q_1_hidden1_common/BiasAddBiasAdd*critic_q_1_hidden1_common/MatMul:product:08critic_q_1_hidden1_common/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!critic_q_1_hidden1_common/BiasAddз
critic_q_1_hidden1_common/ReluRelu*critic_q_1_hidden1_common/BiasAdd:output:0*
T0*(
_output_shapes
:         А2 
critic_q_1_hidden1_common/Relu▌
/critic_q_1_hidden2_common/MatMul/ReadVariableOpReadVariableOp8critic_q_1_hidden2_common_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype021
/critic_q_1_hidden2_common/MatMul/ReadVariableOpш
 critic_q_1_hidden2_common/MatMulMatMul,critic_q_1_hidden1_common/Relu:activations:07critic_q_1_hidden2_common/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 critic_q_1_hidden2_common/MatMul█
0critic_q_1_hidden2_common/BiasAdd/ReadVariableOpReadVariableOp9critic_q_1_hidden2_common_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0critic_q_1_hidden2_common/BiasAdd/ReadVariableOpъ
!critic_q_1_hidden2_common/BiasAddBiasAdd*critic_q_1_hidden2_common/MatMul:product:08critic_q_1_hidden2_common/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!critic_q_1_hidden2_common/BiasAddз
critic_q_1_hidden2_common/ReluRelu*critic_q_1_hidden2_common/BiasAdd:output:0*
T0*(
_output_shapes
:         А2 
critic_q_1_hidden2_common/Relu─
'critic_q_1_output/MatMul/ReadVariableOpReadVariableOp0critic_q_1_output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'critic_q_1_output/MatMul/ReadVariableOp╧
critic_q_1_output/MatMulMatMul,critic_q_1_hidden2_common/Relu:activations:0/critic_q_1_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
critic_q_1_output/MatMul┬
(critic_q_1_output/BiasAdd/ReadVariableOpReadVariableOp1critic_q_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(critic_q_1_output/BiasAdd/ReadVariableOp╔
critic_q_1_output/BiasAddBiasAdd"critic_q_1_output/MatMul:product:00critic_q_1_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
critic_q_1_output/BiasAddv
IdentityIdentity"critic_q_1_output/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         :::::::Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
╦
Б
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_1991372
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisБ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Ж
Й
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991176
critic_q_1_input_states
critic_q_1_input_actions%
!critic_q_1_hidden1_common_1991160%
!critic_q_1_hidden1_common_1991162%
!critic_q_1_hidden2_common_1991165%
!critic_q_1_hidden2_common_1991167
critic_q_1_output_1991170
critic_q_1_output_1991172
identityИв1critic_q_1_hidden1_common/StatefulPartitionedCallв1critic_q_1_hidden2_common/StatefulPartitionedCallв)critic_q_1_output/StatefulPartitionedCall╡
(critic_q_1_concatenation/PartitionedCallPartitionedCallcritic_q_1_input_statescritic_q_1_input_actions*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *^
fYRW
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_19910652*
(critic_q_1_concatenation/PartitionedCallЬ
1critic_q_1_hidden1_common/StatefulPartitionedCallStatefulPartitionedCall1critic_q_1_concatenation/PartitionedCall:output:0!critic_q_1_hidden1_common_1991160!critic_q_1_hidden1_common_1991162*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_199108523
1critic_q_1_hidden1_common/StatefulPartitionedCallе
1critic_q_1_hidden2_common/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden1_common/StatefulPartitionedCall:output:0!critic_q_1_hidden2_common_1991165!critic_q_1_hidden2_common_1991167*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_199111223
1critic_q_1_hidden2_common/StatefulPartitionedCall№
)critic_q_1_output/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden2_common/StatefulPartitionedCall:output:0critic_q_1_output_1991170critic_q_1_output_1991172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *W
fRRP
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_19911382+
)critic_q_1_output/StatefulPartitionedCallЪ
IdentityIdentity2critic_q_1_output/StatefulPartitionedCall:output:02^critic_q_1_hidden1_common/StatefulPartitionedCall2^critic_q_1_hidden2_common/StatefulPartitionedCall*^critic_q_1_output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::2f
1critic_q_1_hidden1_common/StatefulPartitionedCall1critic_q_1_hidden1_common/StatefulPartitionedCall2f
1critic_q_1_hidden2_common/StatefulPartitionedCall1critic_q_1_hidden2_common/StatefulPartitionedCall2V
)critic_q_1_output/StatefulPartitionedCall)critic_q_1_output/StatefulPartitionedCall:` \
'
_output_shapes
:         
1
_user_specified_namecritic_q_1_input_states:a]
'
_output_shapes
:         
2
_user_specified_namecritic_q_1_input_actions
┴
╛
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_1991389

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Г
#__inference__traced_restore_1991507
file_prefix5
1assignvariableop_critic_q_1_hidden1_common_kernel5
1assignvariableop_1_critic_q_1_hidden1_common_bias7
3assignvariableop_2_critic_q_1_hidden2_common_kernel5
1assignvariableop_3_critic_q_1_hidden2_common_bias/
+assignvariableop_4_critic_q_1_output_kernel-
)assignvariableop_5_critic_q_1_output_bias

identity_7ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
valueєBЁB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices╬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity░
AssignVariableOpAssignVariableOp1assignvariableop_critic_q_1_hidden1_common_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1╢
AssignVariableOp_1AssignVariableOp1assignvariableop_1_critic_q_1_hidden1_common_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2╕
AssignVariableOp_2AssignVariableOp3assignvariableop_2_critic_q_1_hidden2_common_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╢
AssignVariableOp_3AssignVariableOp1assignvariableop_3_critic_q_1_hidden2_common_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4░
AssignVariableOp_4AssignVariableOp+assignvariableop_4_critic_q_1_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5о
AssignVariableOp_5AssignVariableOp)assignvariableop_5_critic_q_1_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpф

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6╓

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╒	
Є
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991255
critic_q_1_input_states
critic_q_1_input_actions
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallcritic_q_1_input_statescritic_q_1_input_actionsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8В *V
fQRO
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_19912402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:         
1
_user_specified_namecritic_q_1_input_states:a]
'
_output_shapes
:         
2
_user_specified_namecritic_q_1_input_actions
Э	
х
%__inference_signature_wrapper_1991275
critic_q_1_input_actions
critic_q_1_input_states
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallcritic_q_1_input_statescritic_q_1_input_actionsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8В *+
f&R$
"__inference__wrapped_model_19910532
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
'
_output_shapes
:         
2
_user_specified_namecritic_q_1_input_actions:`\
'
_output_shapes
:         
1
_user_specified_namecritic_q_1_input_states
Й
Р
;__inference_critic_q_1_hidden2_common_layer_call_fn_1991418

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_19911122
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ж
Й
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991155
critic_q_1_input_states
critic_q_1_input_actions%
!critic_q_1_hidden1_common_1991096%
!critic_q_1_hidden1_common_1991098%
!critic_q_1_hidden2_common_1991123%
!critic_q_1_hidden2_common_1991125
critic_q_1_output_1991149
critic_q_1_output_1991151
identityИв1critic_q_1_hidden1_common/StatefulPartitionedCallв1critic_q_1_hidden2_common/StatefulPartitionedCallв)critic_q_1_output/StatefulPartitionedCall╡
(critic_q_1_concatenation/PartitionedCallPartitionedCallcritic_q_1_input_statescritic_q_1_input_actions*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *^
fYRW
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_19910652*
(critic_q_1_concatenation/PartitionedCallЬ
1critic_q_1_hidden1_common/StatefulPartitionedCallStatefulPartitionedCall1critic_q_1_concatenation/PartitionedCall:output:0!critic_q_1_hidden1_common_1991096!critic_q_1_hidden1_common_1991098*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_199108523
1critic_q_1_hidden1_common/StatefulPartitionedCallе
1critic_q_1_hidden2_common/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden1_common/StatefulPartitionedCall:output:0!critic_q_1_hidden2_common_1991123!critic_q_1_hidden2_common_1991125*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_199111223
1critic_q_1_hidden2_common/StatefulPartitionedCall№
)critic_q_1_output/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden2_common/StatefulPartitionedCall:output:0critic_q_1_output_1991149critic_q_1_output_1991151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *W
fRRP
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_19911382+
)critic_q_1_output/StatefulPartitionedCallЪ
IdentityIdentity2critic_q_1_output/StatefulPartitionedCall:output:02^critic_q_1_hidden1_common/StatefulPartitionedCall2^critic_q_1_hidden2_common/StatefulPartitionedCall*^critic_q_1_output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::2f
1critic_q_1_hidden1_common/StatefulPartitionedCall1critic_q_1_hidden1_common/StatefulPartitionedCall2f
1critic_q_1_hidden2_common/StatefulPartitionedCall1critic_q_1_hidden2_common/StatefulPartitionedCall2V
)critic_q_1_output/StatefulPartitionedCall)critic_q_1_output/StatefulPartitionedCall:` \
'
_output_shapes
:         
1
_user_specified_namecritic_q_1_input_states:a]
'
_output_shapes
:         
2
_user_specified_namecritic_q_1_input_actions
┴

U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_1991065

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
б
ш
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991201

inputs
inputs_1%
!critic_q_1_hidden1_common_1991185%
!critic_q_1_hidden1_common_1991187%
!critic_q_1_hidden2_common_1991190%
!critic_q_1_hidden2_common_1991192
critic_q_1_output_1991195
critic_q_1_output_1991197
identityИв1critic_q_1_hidden1_common/StatefulPartitionedCallв1critic_q_1_hidden2_common/StatefulPartitionedCallв)critic_q_1_output/StatefulPartitionedCallФ
(critic_q_1_concatenation/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8В *^
fYRW
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_19910652*
(critic_q_1_concatenation/PartitionedCallЬ
1critic_q_1_hidden1_common/StatefulPartitionedCallStatefulPartitionedCall1critic_q_1_concatenation/PartitionedCall:output:0!critic_q_1_hidden1_common_1991185!critic_q_1_hidden1_common_1991187*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_199108523
1critic_q_1_hidden1_common/StatefulPartitionedCallе
1critic_q_1_hidden2_common/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden1_common/StatefulPartitionedCall:output:0!critic_q_1_hidden2_common_1991190!critic_q_1_hidden2_common_1991192*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_199111223
1critic_q_1_hidden2_common/StatefulPartitionedCall№
)critic_q_1_output/StatefulPartitionedCallStatefulPartitionedCall:critic_q_1_hidden2_common/StatefulPartitionedCall:output:0critic_q_1_output_1991195critic_q_1_output_1991197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *W
fRRP
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_19911382+
)critic_q_1_output/StatefulPartitionedCallЪ
IdentityIdentity2critic_q_1_output/StatefulPartitionedCall:output:02^critic_q_1_hidden1_common/StatefulPartitionedCall2^critic_q_1_hidden2_common/StatefulPartitionedCall*^critic_q_1_output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::2f
1critic_q_1_hidden1_common/StatefulPartitionedCall1critic_q_1_hidden1_common/StatefulPartitionedCall2f
1critic_q_1_hidden2_common/StatefulPartitionedCall1critic_q_1_hidden2_common/StatefulPartitionedCall2V
)critic_q_1_output/StatefulPartitionedCall)critic_q_1_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
ў
И
3__inference_critic_q_1_output_layer_call_fn_1991437

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *W
fRRP
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_19911382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
°
╙
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991347
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*1
config_proto!

CPU

GPU (2J 8В *V
fQRO
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_19912012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
√ 
у
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991302
inputs_0
inputs_1<
8critic_q_1_hidden1_common_matmul_readvariableop_resource=
9critic_q_1_hidden1_common_biasadd_readvariableop_resource<
8critic_q_1_hidden2_common_matmul_readvariableop_resource=
9critic_q_1_hidden2_common_biasadd_readvariableop_resource4
0critic_q_1_output_matmul_readvariableop_resource5
1critic_q_1_output_biasadd_readvariableop_resource
identityИО
$critic_q_1_concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$critic_q_1_concatenation/concat/axis╠
critic_q_1_concatenation/concatConcatV2inputs_0inputs_1-critic_q_1_concatenation/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2!
critic_q_1_concatenation/concat▄
/critic_q_1_hidden1_common/MatMul/ReadVariableOpReadVariableOp8critic_q_1_hidden1_common_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype021
/critic_q_1_hidden1_common/MatMul/ReadVariableOpф
 critic_q_1_hidden1_common/MatMulMatMul(critic_q_1_concatenation/concat:output:07critic_q_1_hidden1_common/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 critic_q_1_hidden1_common/MatMul█
0critic_q_1_hidden1_common/BiasAdd/ReadVariableOpReadVariableOp9critic_q_1_hidden1_common_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0critic_q_1_hidden1_common/BiasAdd/ReadVariableOpъ
!critic_q_1_hidden1_common/BiasAddBiasAdd*critic_q_1_hidden1_common/MatMul:product:08critic_q_1_hidden1_common/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!critic_q_1_hidden1_common/BiasAddз
critic_q_1_hidden1_common/ReluRelu*critic_q_1_hidden1_common/BiasAdd:output:0*
T0*(
_output_shapes
:         А2 
critic_q_1_hidden1_common/Relu▌
/critic_q_1_hidden2_common/MatMul/ReadVariableOpReadVariableOp8critic_q_1_hidden2_common_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype021
/critic_q_1_hidden2_common/MatMul/ReadVariableOpш
 critic_q_1_hidden2_common/MatMulMatMul,critic_q_1_hidden1_common/Relu:activations:07critic_q_1_hidden2_common/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 critic_q_1_hidden2_common/MatMul█
0critic_q_1_hidden2_common/BiasAdd/ReadVariableOpReadVariableOp9critic_q_1_hidden2_common_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0critic_q_1_hidden2_common/BiasAdd/ReadVariableOpъ
!critic_q_1_hidden2_common/BiasAddBiasAdd*critic_q_1_hidden2_common/MatMul:product:08critic_q_1_hidden2_common/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!critic_q_1_hidden2_common/BiasAddз
critic_q_1_hidden2_common/ReluRelu*critic_q_1_hidden2_common/BiasAdd:output:0*
T0*(
_output_shapes
:         А2 
critic_q_1_hidden2_common/Relu─
'critic_q_1_output/MatMul/ReadVariableOpReadVariableOp0critic_q_1_output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'critic_q_1_output/MatMul/ReadVariableOp╧
critic_q_1_output/MatMulMatMul,critic_q_1_hidden2_common/Relu:activations:0/critic_q_1_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
critic_q_1_output/MatMul┬
(critic_q_1_output/BiasAdd/ReadVariableOpReadVariableOp1critic_q_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(critic_q_1_output/BiasAdd/ReadVariableOp╔
critic_q_1_output/BiasAddBiasAdd"critic_q_1_output/MatMul:product:00critic_q_1_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
critic_q_1_output/BiasAddv
IdentityIdentity"critic_q_1_output/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         :::::::Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
к)
╜
"__inference__wrapped_model_1991053
critic_q_1_input_states
critic_q_1_input_actionsM
Icritic_q_1_model_critic_q_1_hidden1_common_matmul_readvariableop_resourceN
Jcritic_q_1_model_critic_q_1_hidden1_common_biasadd_readvariableop_resourceM
Icritic_q_1_model_critic_q_1_hidden2_common_matmul_readvariableop_resourceN
Jcritic_q_1_model_critic_q_1_hidden2_common_biasadd_readvariableop_resourceE
Acritic_q_1_model_critic_q_1_output_matmul_readvariableop_resourceF
Bcritic_q_1_model_critic_q_1_output_biasadd_readvariableop_resource
identityИ░
5CRITIC_Q_1_MODEL/critic_q_1_concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :27
5CRITIC_Q_1_MODEL/critic_q_1_concatenation/concat/axisЮ
0CRITIC_Q_1_MODEL/critic_q_1_concatenation/concatConcatV2critic_q_1_input_statescritic_q_1_input_actions>CRITIC_Q_1_MODEL/critic_q_1_concatenation/concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
0CRITIC_Q_1_MODEL/critic_q_1_concatenation/concatП
@CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/MatMul/ReadVariableOpReadVariableOpIcritic_q_1_model_critic_q_1_hidden1_common_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02B
@CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/MatMul/ReadVariableOpи
1CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/MatMulMatMul9CRITIC_Q_1_MODEL/critic_q_1_concatenation/concat:output:0HCRITIC_Q_1_MODEL/critic_q_1_hidden1_common/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А23
1CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/MatMulО
ACRITIC_Q_1_MODEL/critic_q_1_hidden1_common/BiasAdd/ReadVariableOpReadVariableOpJcritic_q_1_model_critic_q_1_hidden1_common_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02C
ACRITIC_Q_1_MODEL/critic_q_1_hidden1_common/BiasAdd/ReadVariableOpо
2CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/BiasAddBiasAdd;CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/MatMul:product:0ICRITIC_Q_1_MODEL/critic_q_1_hidden1_common/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А24
2CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/BiasAdd┌
/CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/ReluRelu;CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/BiasAdd:output:0*
T0*(
_output_shapes
:         А21
/CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/ReluР
@CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/MatMul/ReadVariableOpReadVariableOpIcritic_q_1_model_critic_q_1_hidden2_common_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02B
@CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/MatMul/ReadVariableOpм
1CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/MatMulMatMul=CRITIC_Q_1_MODEL/critic_q_1_hidden1_common/Relu:activations:0HCRITIC_Q_1_MODEL/critic_q_1_hidden2_common/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А23
1CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/MatMulО
ACRITIC_Q_1_MODEL/critic_q_1_hidden2_common/BiasAdd/ReadVariableOpReadVariableOpJcritic_q_1_model_critic_q_1_hidden2_common_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02C
ACRITIC_Q_1_MODEL/critic_q_1_hidden2_common/BiasAdd/ReadVariableOpо
2CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/BiasAddBiasAdd;CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/MatMul:product:0ICRITIC_Q_1_MODEL/critic_q_1_hidden2_common/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А24
2CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/BiasAdd┌
/CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/ReluRelu;CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/BiasAdd:output:0*
T0*(
_output_shapes
:         А21
/CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/Reluў
8CRITIC_Q_1_MODEL/critic_q_1_output/MatMul/ReadVariableOpReadVariableOpAcritic_q_1_model_critic_q_1_output_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02:
8CRITIC_Q_1_MODEL/critic_q_1_output/MatMul/ReadVariableOpУ
)CRITIC_Q_1_MODEL/critic_q_1_output/MatMulMatMul=CRITIC_Q_1_MODEL/critic_q_1_hidden2_common/Relu:activations:0@CRITIC_Q_1_MODEL/critic_q_1_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)CRITIC_Q_1_MODEL/critic_q_1_output/MatMulї
9CRITIC_Q_1_MODEL/critic_q_1_output/BiasAdd/ReadVariableOpReadVariableOpBcritic_q_1_model_critic_q_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9CRITIC_Q_1_MODEL/critic_q_1_output/BiasAdd/ReadVariableOpН
*CRITIC_Q_1_MODEL/critic_q_1_output/BiasAddBiasAdd3CRITIC_Q_1_MODEL/critic_q_1_output/MatMul:product:0ACRITIC_Q_1_MODEL/critic_q_1_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*CRITIC_Q_1_MODEL/critic_q_1_output/BiasAddЗ
IdentityIdentity3CRITIC_Q_1_MODEL/critic_q_1_output/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:         :         :::::::` \
'
_output_shapes
:         
1
_user_specified_namecritic_q_1_input_states:a]
'
_output_shapes
:         
2
_user_specified_namecritic_q_1_input_actions
┤
▌
 __inference__traced_save_1991479
file_prefix?
;savev2_critic_q_1_hidden1_common_kernel_read_readvariableop=
9savev2_critic_q_1_hidden1_common_bias_read_readvariableop?
;savev2_critic_q_1_hidden2_common_kernel_read_readvariableop=
9savev2_critic_q_1_hidden2_common_bias_read_readvariableop7
3savev2_critic_q_1_output_kernel_read_readvariableop5
1savev2_critic_q_1_output_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_30fbaa7f66564dea9f9c67442b7f247f/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameы
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
valueєBЁB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЦ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slicesШ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_critic_q_1_hidden1_common_kernel_read_readvariableop9savev2_critic_q_1_hidden1_common_bias_read_readvariableop;savev2_critic_q_1_hidden2_common_kernel_read_readvariableop9savev2_critic_q_1_hidden2_common_bias_read_readvariableop3savev2_critic_q_1_output_kernel_read_readvariableop1savev2_critic_q_1_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*M
_input_shapes<
:: :	А:А:
АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: 
┴
╛
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_1991085

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З
Р
;__inference_critic_q_1_hidden1_common_layer_call_fn_1991398

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8В *_
fZRX
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_19910852
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
]
critic_q_1_input_actionsA
*serving_default_critic_q_1_input_actions:0         
[
critic_q_1_input_states@
)serving_default_critic_q_1_input_states:0         E
critic_q_1_output0
StatefulPartitionedCall:0         tensorflow/serving/predict:ьз
ў.
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
#_self_saveable_object_factories

signatures
	regularization_losses

trainable_variables
	variables
	keras_api
B__call__
C_default_save_signature
*D&call_and_return_all_conditional_losses"·+
_tf_keras_network▐+{"class_name": "Functional", "name": "CRITIC_Q_1_MODEL", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CRITIC_Q_1_MODEL", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "critic_q_1_input_states"}, "name": "critic_q_1_input_states", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "critic_q_1_input_actions"}, "name": "critic_q_1_input_actions", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "critic_q_1_concatenation", "trainable": true, "dtype": "float32", "axis": -1}, "name": "critic_q_1_concatenation", "inbound_nodes": [[["critic_q_1_input_states", 0, 0, {}], ["critic_q_1_input_actions", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "critic_q_1_hidden1_common", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "critic_q_1_hidden1_common", "inbound_nodes": [[["critic_q_1_concatenation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "critic_q_1_hidden2_common", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "critic_q_1_hidden2_common", "inbound_nodes": [[["critic_q_1_hidden1_common", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "critic_q_1_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "critic_q_1_output", "inbound_nodes": [[["critic_q_1_hidden2_common", 0, 0, {}]]]}], "input_layers": [["critic_q_1_input_states", 0, 0], ["critic_q_1_input_actions", 0, 0]], "output_layers": [["critic_q_1_output", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 24]}, {"class_name": "TensorShape", "items": [null, 4]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CRITIC_Q_1_MODEL", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "critic_q_1_input_states"}, "name": "critic_q_1_input_states", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "critic_q_1_input_actions"}, "name": "critic_q_1_input_actions", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "critic_q_1_concatenation", "trainable": true, "dtype": "float32", "axis": -1}, "name": "critic_q_1_concatenation", "inbound_nodes": [[["critic_q_1_input_states", 0, 0, {}], ["critic_q_1_input_actions", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "critic_q_1_hidden1_common", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "critic_q_1_hidden1_common", "inbound_nodes": [[["critic_q_1_concatenation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "critic_q_1_hidden2_common", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "critic_q_1_hidden2_common", "inbound_nodes": [[["critic_q_1_hidden1_common", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "critic_q_1_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "critic_q_1_output", "inbound_nodes": [[["critic_q_1_hidden2_common", 0, 0, {}]]]}], "input_layers": [["critic_q_1_input_states", 0, 0], ["critic_q_1_input_actions", 0, 0]], "output_layers": [["critic_q_1_output", 0, 0]]}}}
░
#_self_saveable_object_factories"И
_tf_keras_input_layerш{"class_name": "InputLayer", "name": "critic_q_1_input_states", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "critic_q_1_input_states"}}
░
#_self_saveable_object_factories"И
_tf_keras_input_layerш{"class_name": "InputLayer", "name": "critic_q_1_input_actions", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "critic_q_1_input_actions"}}
З
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"╙
_tf_keras_layer╣{"class_name": "Concatenate", "name": "critic_q_1_concatenation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "critic_q_1_concatenation", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 24]}, {"class_name": "TensorShape", "items": [null, 4]}]}
║

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
G__call__
*H&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"class_name": "Dense", "name": "critic_q_1_hidden1_common", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "critic_q_1_hidden1_common", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28]}}
╝

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
 	variables
!	keras_api
I__call__
*J&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "Dense", "name": "critic_q_1_hidden2_common", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "critic_q_1_hidden2_common", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
м

"kernel
#bias
#$_self_saveable_object_factories
%regularization_losses
&trainable_variables
'	variables
(	keras_api
K__call__
*L&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "Dense", "name": "critic_q_1_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "critic_q_1_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_dict_wrapper
,
Mserving_default"
signature_map
 "
trackable_list_wrapper
J
0
1
2
3
"4
#5"
trackable_list_wrapper
J
0
1
2
3
"4
#5"
trackable_list_wrapper
╩
)non_trainable_variables
	regularization_losses
*metrics
+layer_metrics

,layers

trainable_variables
	variables
-layer_regularization_losses
B__call__
C_default_save_signature
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
.non_trainable_variables
/metrics
regularization_losses
0layer_metrics

1layers
trainable_variables
	variables
2layer_regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
3:1	А2 critic_q_1_hidden1_common/kernel
-:+А2critic_q_1_hidden1_common/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
3non_trainable_variables
4metrics
regularization_losses
5layer_metrics

6layers
trainable_variables
	variables
7layer_regularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
4:2
АА2 critic_q_1_hidden2_common/kernel
-:+А2critic_q_1_hidden2_common/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
8non_trainable_variables
9metrics
regularization_losses
:layer_metrics

;layers
trainable_variables
 	variables
<layer_regularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
+:)	А2critic_q_1_output/kernel
$:"2critic_q_1_output/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
н
=non_trainable_variables
>metrics
%regularization_losses
?layer_metrics

@layers
&trainable_variables
'	variables
Alayer_regularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ц2У
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991365
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991347
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991216
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991255└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
й2ж
"__inference__wrapped_model_1991053 
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *oвl
jЪg
1К.
critic_q_1_input_states         
2К/
critic_q_1_input_actions         
В2 
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991329
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991302
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991155
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991176└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ф2с
:__inference_critic_q_1_concatenation_layer_call_fn_1991378в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 2№
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_1991372в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т
;__inference_critic_q_1_hidden1_common_layer_call_fn_1991398в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
А2¤
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_1991389в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т
;__inference_critic_q_1_hidden2_common_layer_call_fn_1991418в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
А2¤
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_1991409в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▌2┌
3__inference_critic_q_1_output_layer_call_fn_1991437в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°2ї
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_1991428в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
\BZ
%__inference_signature_wrapper_1991275critic_q_1_input_actionscritic_q_1_input_statesЕ
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991155│"#Бв~
wвt
jЪg
1К.
critic_q_1_input_states         
2К/
critic_q_1_input_actions         
p

 
к "%в"
К
0         
Ъ Е
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991176│"#Бв~
wвt
jЪg
1К.
critic_q_1_input_states         
2К/
critic_q_1_input_actions         
p 

 
к "%в"
К
0         
Ъ х
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991302У"#bв_
XвU
KЪH
"К
inputs/0         
"К
inputs/1         
p

 
к "%в"
К
0         
Ъ х
M__inference_CRITIC_Q_1_MODEL_layer_call_and_return_conditional_losses_1991329У"#bв_
XвU
KЪH
"К
inputs/0         
"К
inputs/1         
p 

 
к "%в"
К
0         
Ъ ▌
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991216ж"#Бв~
wвt
jЪg
1К.
critic_q_1_input_states         
2К/
critic_q_1_input_actions         
p

 
к "К         ▌
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991255ж"#Бв~
wвt
jЪg
1К.
critic_q_1_input_states         
2К/
critic_q_1_input_actions         
p 

 
к "К         ╜
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991347Ж"#bв_
XвU
KЪH
"К
inputs/0         
"К
inputs/1         
p

 
к "К         ╜
2__inference_CRITIC_Q_1_MODEL_layer_call_fn_1991365Ж"#bв_
XвU
KЪH
"К
inputs/0         
"К
inputs/1         
p 

 
к "К         ё
"__inference__wrapped_model_1991053╩"#yвv
oвl
jЪg
1К.
critic_q_1_input_states         
2К/
critic_q_1_input_actions         
к "EкB
@
critic_q_1_output+К(
critic_q_1_output         ▌
U__inference_critic_q_1_concatenation_layer_call_and_return_conditional_losses_1991372ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "%в"
К
0         
Ъ ┤
:__inference_critic_q_1_concatenation_layer_call_fn_1991378vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "К         ╖
V__inference_critic_q_1_hidden1_common_layer_call_and_return_conditional_losses_1991389]/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ П
;__inference_critic_q_1_hidden1_common_layer_call_fn_1991398P/в,
%в"
 К
inputs         
к "К         А╕
V__inference_critic_q_1_hidden2_common_layer_call_and_return_conditional_losses_1991409^0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ Р
;__inference_critic_q_1_hidden2_common_layer_call_fn_1991418Q0в-
&в#
!К
inputs         А
к "К         Ап
N__inference_critic_q_1_output_layer_call_and_return_conditional_losses_1991428]"#0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ З
3__inference_critic_q_1_output_layer_call_fn_1991437P"#0в-
&в#
!К
inputs         А
к "К         к
%__inference_signature_wrapper_1991275А"#овк
в 
вкЮ
N
critic_q_1_input_actions2К/
critic_q_1_input_actions         
L
critic_q_1_input_states1К.
critic_q_1_input_states         "EкB
@
critic_q_1_output+К(
critic_q_1_output         