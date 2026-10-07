# Workspace IAM AssumeRole

Ce workspace introduit la delegation cross-account pour un service ou un pipeline GenAI.

## Ce que vous y cherchez
- la trust policy qui autorise l'assomption du role
- la permissions policy qui limite l'action utile
- les garde-fous comme ExternalId et la portee minimale des droits

## Piste de depart
Commencez par lire la trust policy avant la permissions policy: la premiere decide qui peut entrer.
