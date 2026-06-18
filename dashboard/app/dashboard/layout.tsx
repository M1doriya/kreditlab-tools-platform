import { ReactNode } from "react";
import { Sidebar } from "./sidebar";
export default function DashboardLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="flex h-dvh overflow-hidden bg-slate-100">
      <Sidebar />
      <main className="min-w-0 flex-1 overflow-y-auto overflow-x-hidden">
        {children}
      </main>
    </div>
  );
}
