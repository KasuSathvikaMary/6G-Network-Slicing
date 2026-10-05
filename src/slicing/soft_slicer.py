import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResourceLoan:
    borrower: str
    lender: str
    amount: float
    expiry: float

class SoftSlicer:
    def __init__(self):
        self.resources: Dict[str, float] = {}
        self.loans: List[ResourceLoan] = []
        self.borrow_limit = 0.3

    def get_sharing_status(self) -> Dict:
        return {"resources": self.resources, "loans": self.loans}

    def borrow_resources(self, borrower: str, lender: str, amount: float, duration: float) -> bool:
        if lender not in self.resources or borrower not in self.resources:
            return False
        max_borrow = self.resources[lender] * self.borrow_limit
        if amount > max_borrow:
            return False
        
        loan = ResourceLoan(borrower, lender, amount, time.time() + duration)
        self.loans.append(loan)
        self.resources[lender] -= amount
        self.resources[borrower] += amount
        logger.info(f"{borrower} borrowed {amount} from {lender}")
        return True

    def return_resources(self, borrower: str, lender: str, amount: float) -> None:
        self.resources[borrower] -= amount
        self.resources[lender] += amount
        self.loans = [l for l in self.loans if not (l.borrower == borrower and l.lender == lender and l.amount == amount)]

    def rebalance(self) -> None:
        current_time = time.time()
        expired = [l for l in self.loans if l.expiry <= current_time]
        for loan in expired:
            self.return_resources(loan.borrower, loan.lender, loan.amount)
