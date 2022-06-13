(define (domain bi-panda-tamp)
  (:requirements :strips :equality)
  (:constants @sink @stove)
  (:predicates
    (Arm ?a)
    (Stackable ?o ?r)
    (Sink ?r)
    (Stove ?r)
    (Type ?t ?b)

    (Pose ?o ?p)
    (Grasp ?o ?g)
    (Kin ?a ?o ?p ?g ?q ?t)
    (ArmMotion ?a ?q1 ?t ?q2)
    (Supported ?o ?p ?r)
    (BTraj ?t)
    (ATraj ?t)
    (ForcesBalanced o? p?)

    (CFreePosePose ?o ?p ?o2 ?p2)
    (CFreeApproachPose ?o ?p ?g ?o2 ?p2)
    (CFreeTrajPose ?t ?o2 ?p2)
    (CFreeTrajGraspPose ?t ?a ?o1 ?g1 ?o2 ?p2)

    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (HandEmpty ?a)
    (AtBConf ?q)
    (AtAConf ?a ?q)
    (CanMove)
    (Cleaned ?o)
    (Cooked ?o)

    (On ?o ?r)
    (Holding ?a ?o)
    (UnsafePose ?o ?p)
    (UnsafeApproach ?o ?p ?g)
    (UnsafeATraj ?t)
    (UnsafeBTraj ?t)
  )
  (:functions
    (MoveCost ?t)
    (PickCost)
    (PlaceCost)
  )

  ;(:action move_arm
  ;  :parameters (?q1 ?q2 ?t)
  ;  :precondition (and (ArmMotion ?a ?q1 ?t ?q2)
  ;                     (AtAConf ?a ?q1))
  ;  :effect (and (AtAConf ?a ?q2)
  ;               (not (AtAConf ?a ?q1)))
  ;)

  (:action pick
    :parameters (?a ?o ?p ?g ?q ?t)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t)
                       (AtPose ?o ?p) (HandEmpty ?a)
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeATraj ?t))
                  )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove)
                 (not (AtPose ?o ?p)) (not (HandEmpty ?a))
                 (increase (total-cost) (PickCost)))
  )
  (:action place
    :parameters (?a ?o ?p ?g ?q ?t ?r)
    :precondition (and (Kin ?a ?o ?p ?g ?q ?t)
                        (ForcesBalanced ?o ?p ?r)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafeApproach ?o ?p ?g))
                       (not (UnsafeATraj ?t))
                  )
    :effect (and (AtPose ?o ?p) (HandEmpty ?a) (CanMove)
                 (not (AtGrasp ?a ?o ?g)) (On ?o ?r)
                 (increase (total-cost) (PlaceCost)))
  )

  (:action clean
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Type ?r @sink) ; (Sink ?r)
                       (On ?o ?r))
    :effect (Cleaned ?o)
  )
  (:action cook
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Type ?r @stove) ; (Stove ?r)
                       (On ?o ?r) (Cleaned ?o))
    :effect (and (Cooked ?o)
                 (not (Cleaned ?o)))
  )

  (:derived (On ?o ?r)
    (exists (?p) (and (Supported ?o ?p ?r)
                      (AtPose ?o ?p)))
  )
  (:derived (Holding ?a ?o)
    (exists (?g) (and (Arm ?a) (Grasp ?o ?g)
                      (AtGrasp ?a ?o ?g)))
  )

  (:derived (UnsafePose ?o ?p)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreePosePose ?o ?p ?o2 ?p2))
                           (AtPose ?o2 ?p2))
  )
  (:derived (UnsafeApproach ?o ?p ?g)
    (exists (?o2 ?p2) (and (Pose ?o ?p) (Grasp ?o ?g) (Pose ?o2 ?p2) (not (= ?o ?o2))
                           (not (CFreeApproachPose ?o ?p ?g ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )

  (:derived (UnsafeATraj ?t)
    (exists (?o2 ?p2) (and (ATraj ?t) (Pose ?o2 ?p2)
                           (not (CFreeTrajPose ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)))
  )
  ;(:derived (UnsafeBTraj ?t) (or
  ;  ;(exists (?o2 ?p2) (and (BTraj ?t) (Pose ?o2 ?p2)
  ;  ;                       (not (CFreeTrajPose ?t ?o2 ?p2))
  ;  ;                       (AtPose ?o2 ?p2)))
  ;  (exists (?a ?o1 ?g1 ?o2 ?p2) (and (BTraj ?t) (Arm ?a) (Grasp ?o1 ?g1) (Pose ?o2 ?p2)
  ;                                    (not (CFreeTrajGraspPose ?t ?a ?o1 ?g1 ?o2 ?p2)) (not (= ?o1 ?o2))
  ;                                    (AtGrasp ?a ?o1 ?g1) (AtPose ?o2 ?p2)))
  ;))

)