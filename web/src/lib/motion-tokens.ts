// Motion design tokens for fluid spring transitions (EasyUI / Framer Motion)
export const motionTransitions = {
  springGentle: {
    type: 'spring',
    stiffness: 300,
    damping: 30,
  },
  springMorph: {
    type: 'spring',
    stiffness: 400,
    damping: 35,
  },
  springBouncy: {
    type: 'spring',
    stiffness: 500,
    damping: 25,
  },
  fadeFast: {
    duration: 0.15,
    ease: 'easeOut',
  },
};
