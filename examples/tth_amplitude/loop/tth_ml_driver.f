C     Batch driver for the standalone MadLoop process: reads one phase-space
C     point per line from stdin (E px py pz for each external particle, GeV,
C     in the process order) and prints born, finite, 1/eps and 1/eps^2
C     coefficients (the latter three without the alpha_s/(2 pi) factor), the
C     estimated relative accuracy and the MadLoop return code; mu_R = sqrt(s)/2.
      PROGRAM DRIVER
      IMPLICIT NONE
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      INTEGER NEXTERNAL, NINCOMING
      PARAMETER (NEXTERNAL=5,NINCOMING=2)
      INCLUDE 'coupl.inc'
      REAL*8 PMASS(NEXTERNAL)
      INTEGER I,J
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER MATELEM_ARRAY_DIM
      REAL*8 , ALLOCATABLE :: MATELEM(:,:)
      REAL*8 SQRTS,AO2PI
      REAL*8 PIN(0:3)
      INTEGER RETURNCODE
      INTEGER NSQUAREDSO_LOOP
      REAL*8 , ALLOCATABLE :: PREC_FOUND(:)
      REAL*8 DOT
      EXTERNAL DOT
      CALL ML5_0_GET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
      ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))
      CALL ML5_0_GET_NSQSO_LOOP(NSQUAREDSO_LOOP)
      ALLOCATE(PREC_FOUND(0:NSQUAREDSO_LOOP))
      CALL SETPARA('param_card.dat')
      INCLUDE 'pmass.inc'
 10   CONTINUE
      READ(*,*,END=999) ((P(I,J),I=0,3),J=1,NEXTERNAL)
      DO I=0,3
        PIN(I)=0.0D0
        DO J=1,NINCOMING
          PIN(I)=PIN(I)+P(I,J)
        ENDDO
      ENDDO
      SQRTS=DSQRT(DABS(DOT(PIN(0),PIN(0))))
      MU_R=SQRTS/2D0
      CALL UPDATE_AS_PARAM()
      AO2PI=G**2/(8.D0*(3.14159265358979323846D0**2))
      CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,-1.0D0,PREC_FOUND,
     $ RETURNCODE)
      WRITE(*,'(A4,4ES24.15,ES12.3,I6)') 'RES ', MATELEM(0,0), MATELEM(1,0)/AO2PI,
     $ MATELEM(2,0)/AO2PI, MATELEM(3,0)/AO2PI, PREC_FOUND(0), RETURNCODE
      CALL FLUSH(6)
      GOTO 10
 999  CONTINUE
      END

      REAL*8 FUNCTION DOT(P1,P2)
      IMPLICIT NONE
      REAL*8 P1(0:3),P2(0:3)
      DOT=P1(0)*P2(0)-P1(1)*P2(1)-P1(2)*P2(2)-P1(3)*P2(3)
      END
