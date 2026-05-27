from agents.c51 import C51Agent
from agents.codac import CODACAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.iqn import IQNAgent
from agents.lambda_flow import LambdaFlowAgent
from agents.rebrac import ReBRACAgent
from agents.value_flows import ValueFlowsAgent

agents = dict(
    c51=C51Agent,
    codac=CODACAgent,
    fbrac=FBRACAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    iqn=IQNAgent,
    lambda_flow=LambdaFlowAgent,
    rebrac=ReBRACAgent,
    value_flows=ValueFlowsAgent,
)