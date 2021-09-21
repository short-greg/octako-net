from takonet.machinery.learners import Learner
from takonet.teaching.bulletins import Bulletin, BulletinAccessor, BulletinController, Course, Entry, EntryAccessor, RunProgress, StatusLog, TeacherBulletin
import pytest


class TestRunProgress:

    def test_cur_iteration_after_reset_equals_0(self):

        rp = RunProgress(2, 2)
        rp.cur_iteration += 1
        rp.reset()
        assert rp.cur_iteration == 0


    def test_cur_round_after_reset_equals_0(self):

        rp = RunProgress(2, 2)
        rp.cur_round += 1
        rp.reset()
        assert rp.cur_round == 0


class TestEntry:

    def test_results_after_append_one_results_is_correct(self):
        entry = Entry(1, 1, 1, progress=RunProgress(3, 3))
        entry.append_results({'X': 2, 'Y': 3})
        assert entry.results.loc[0, 'X'] == 2
        assert entry.results.loc[0, 'Y'] == 3

    def test_results_after_reset_is_empty(self):
        entry = Entry(1, 1, 1, progress=RunProgress(3, 3))
        entry.append_results({'X': 2, 'Y': 3})
        entry.reset()
        assert len(entry.results) == 0

    def test_reset_after_finish_doesnt_work(self):
        entry = Entry(1, 1, 1, progress=RunProgress(3, 3))
        entry.finish()
        assert entry.reset() is False

    def test_results_after_append_two_results_is_correct(self):
        entry = Entry(1, 1, 1, progress=RunProgress(3, 3))
        entry.append_results({'X': 2, 'Y': 3})
        entry.append_results({'Z': 4, 'X': 9})
        assert entry.results.loc[0, 'X'] == 2
        assert entry.results.loc[1, 'X'] == 9
        assert entry.results.loc[1, 'Z'] == 4

    def test_is_finished_after_set_finished(self):
        entry = Entry(1, 1, 1, progress=RunProgress(3, 3))
        entry.finish()
        assert entry.is_finished is True

    def test_is_finished_without_set_finished(self):
        entry = Entry(1, 1, 1, progress=RunProgress(3, 3))
        assert entry.is_finished is False


class TestBulletin:

    def test_get_entry_returns_new_entry_for_different_student(self):

        status_log = Bulletin()
        old_entry = status_log.get_entry('teacher', 0)
        entry = status_log.get_entry('teacher', 1)
        assert entry is not old_entry

    def test_get_entry_doesnt_return_new_entry(self):

        status_log = Bulletin()
        old_entry = status_log.get_entry('teacher', 0)
        entry = status_log.get_entry('teacher', 0)
        assert entry is old_entry

    def test_get_entry_returns_current_entry_with_correct_rounds(self):

        status_log = Bulletin()
        entry = status_log.get_entry('teacher', 0)
        entry.progress.n_rounds =  2
        entry = status_log.get_entry('teacher', 0)
        assert entry.progress.n_rounds == 2

    def test_get_entry_returns_new_entry_after_finish(self):

        status_log = Bulletin()
        old_entry = status_log.get_entry('teacher', 0)
        old_entry.finish()
        entry = status_log.get_entry('teacher', 0)
        assert entry is not old_entry


class TestEntryAccessor:

    def test_start_round_increments_round(self):

        entry = Entry('entry', 0, 0, RunProgress())
        accessor = EntryAccessor('entry', entry)
        accessor.start_round()
        assert entry.progress.cur_round == 1

    def test_advance_increments_iteration(self):

        entry = Entry('entry', 0, 0, RunProgress())
        accessor = EntryAccessor('entry', entry)
        accessor.advance()
        assert entry.progress.cur_iteration == 1
        assert entry.progress.cur_round_iteration == 1

    def test_finish_finishes_entry(self):

        entry = Entry('entry', 0, 0, RunProgress())
        accessor = EntryAccessor('entry', entry)
        accessor.finish()
        assert entry.is_finished == True

    def test_cant_increment_after_finish(self):

        entry = Entry('entry', 0, 0, RunProgress())
        accessor = EntryAccessor('entry', entry)
        accessor.finish()
        assert accessor.advance() is False

    def test_update_results_for_entry(self):

        entry = Entry('entry', 0, 0, RunProgress())
        accessor = EntryAccessor('entry', entry)
        accessor.update_results({'T': 4, 'J': 2})
        assert accessor.results.loc[0, 'T'] == 4
        assert accessor.results.loc[0, 'J'] == 2

    def test_update_results_twice_for_entry(self):

        entry = Entry('entry', 0, 0, RunProgress())
        accessor = EntryAccessor('entry', entry)
        accessor.update_results({'T': 4, 'J': 2})
        accessor.update_results({'T': 5, 'J': 3})
        assert accessor.results.loc[0, 'T'] == 4
        assert accessor.results.loc[0, 'J'] == 2
        assert accessor.results.loc[1, 'T'] == 5
        assert accessor.results.loc[1, 'J'] == 3

    def test_reset_resets_progress(self):
        accessor = EntryAccessor('entry', Entry('entry', 0, 0, RunProgress()))
        accessor.advance()
        accessor.reset()
        assert accessor.progress.cur_iteration == 0
        assert accessor.progress.cur_round_iteration == 0

    def test_reset_resets_results(self):
        accessor = EntryAccessor('entry', Entry('entry', 0, 0, RunProgress()))
        accessor.advance()
        accessor.reset()
        assert accessor.results.empty is True


class TestBulletinAccessor:

    def test_teacher_bulletin_gets_new_accessor(self):

        teacher_bulletin = BulletinAccessor('teacher', StatusLog())
        accessor1 = teacher_bulletin.get_entry_accessor(0)
        accessor2 = teacher_bulletin.get_entry_accessor(1)
        accessor1.advance()
        assert accessor1.progress.cur_iteration != accessor2.progress.cur_iteration

    def test_teacher_bulletin_gets_same_accessor(self):

        teacher_bulletin = BulletinAccessor('teacher', StatusLog())
        accessor1 = teacher_bulletin.get_entry_accessor(0)
        accessor2 = teacher_bulletin.get_entry_accessor(0)
        accessor1.advance()
        assert accessor1.progress.cur_iteration == accessor2.progress.cur_iteration

    def test_teacher_bulletin_gets_new_accessor_after_finish(self):

        teacher_bulletin = BulletinAccessor('teacher', StatusLog())
        accessor1 = teacher_bulletin.get_entry_accessor(0)
        accessor1.advance()
        accessor1.finish()
        accessor2 = teacher_bulletin.get_entry_accessor(0)
        assert accessor1.progress.cur_iteration != accessor2.progress.cur_iteration


class LearnerTest(Learner):

    def test(self, x, y):
        pass

    def learn(self, x, y):
        pass


class DummyGoal:

    def evaluate(self, student_id: int, bulletin: Bulletin) -> float:
        return 0.0


class TestCourse:

    def test_register_student(self):

        learner = LearnerTest()
        bulletin = Course(DummyGoal(), 1)
        student = bulletin.register_student(learner)
        assert 0 <= student.id <= 10

    def test_get_teacher_bulletin_returns_teacher_bulletin(self):

        bulletin = Course(DummyGoal(), 1)
        teacher_bulletin = bulletin.get_teacher_bulletin('teacher')
        assert type(teacher_bulletin) is TeacherBulletin

    def test_advance_section_creates_new_status(self):

        bulletin = Course(DummyGoal(), 1)
        teacher_bulletin = bulletin.get_teacher_bulletin('teacher')
        accessor1 = teacher_bulletin.get_entry_accessor(0)
        accessor1.advance()
        bulletin.advance()
        accessor2 = teacher_bulletin.get_entry_accessor(0)

        assert accessor1.progress.cur_iteration != accessor2.progress.cur_iteration
